# docs and experiment results can be found at
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import copy
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from barrier_docking import train_controller_and_certificate,verification_loop
from docking_ppo_presets import apply_preset
from thermometer import ThermometerUniform
from wnn_models import WNN


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "ppo_docking"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = True
    hf_entity: str = ""

    # Algorithm specific arguments
    env_id: str = "Docking2d"
    total_timesteps: int = 6_000_000
    learning_rate: float = 5e-5
    num_envs: int = 4
    num_steps: int = 1000
    anneal_lr: bool = False
    gamma: float = 0.968559
    gae_lambda: float = 0.928544
    num_minibatches: int = 32
    update_epochs: int = 30
    norm_adv: bool = True
    clip_coef: float = 0.3
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.01
    adam_eps: float = 1e-5
    eval_episodes: int = 10
    final_eval_episodes: int = 20
    eval_every_iterations: int = 0
    use_obs_norm: bool = False
    obs_norm_load_path: str = ""
    obs_norm_save_name: str = "obs_norm.pt"
    use_tanh_final: bool = False
    use_discretizer: bool = False
    # Runtime fields
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    # Network and docking preset
    network_type: Literal["relu", "tanh", "wnn", "lgn"] = "wnn"
    preset: Literal["none", "docking2d_saferl"] = "docking2d_saferl"
    float_size: int = 64
    float_layers: int = 2
    float_std: float = 0.01
    save_path: str = "models_ppo_2704"

    # WNN / LGN
    size: int = 2048
    bits: int = 100
    n: int = 2
    l: int = 2
    init_log_alpha: float = -0.6931
    wnn_xy_fov: float = 6
    wnn_vel_fov: float = 0.5
    use_obs_norm_for_wnn: bool = False
    freeze_wnn_interconnect: bool = False
    freeze_wnn_interconnect_after_frac: float = 0.8
    

class ObsNorm(nn.Module):
    def __init__(self, shape, device, eps=1e-8):
        super().__init__()
        mean = torch.zeros(shape, device=device)
        var = torch.ones(shape, device=device)
        count = torch.tensor(eps, device=device)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)
        self.register_buffer("count", count)

    @torch.no_grad()
    def update(self, x):
        b = torch.as_tensor(x, device=self.mean.device, dtype=torch.float32)
        if b.ndim == 1:
            b = b.unsqueeze(0)
        batch_mean = b.mean(0)
        batch_var = b.var(0, unbiased=False)
        batch_count = torch.tensor(b.shape[0], device=self.mean.device, dtype=torch.float32)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def norm(self, x):
        x = torch.as_tensor(x, device=self.mean.device, dtype=torch.float32)
        x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(x, -10, 10)


def _ensure_saferl_on_path():
    try:
        import saferl  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = os.path.dirname(os.path.abspath(__file__))
    saferl_root = os.path.join(repo_root, "SafeRL")
    if os.path.isdir(saferl_root) and saferl_root not in sys.path:
        sys.path.insert(0, saferl_root)


class LegacyGymEnvAdapter(gym.Env):
    """Adapt legacy gym API (obs, reward, done, info) to gymnasium API."""

    def __init__(self, legacy_env):
        super().__init__()
        self.legacy_env = legacy_env
        self._action_unflatten = None
        self.action_space = self._to_gymnasium_action_space(legacy_env.action_space)
        self.observation_space = self._to_gymnasium_space(legacy_env.observation_space)
        self.metadata = getattr(legacy_env, "metadata", {})
        self.reward_range = getattr(legacy_env, "reward_range", (-float("inf"), float("inf")))
        self.spec = getattr(legacy_env, "spec", None)

    @staticmethod
    def _to_gymnasium_space(space):
        if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape"):
            return gym.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        if hasattr(space, "n") and not hasattr(space, "shape"):
            return gym.spaces.Discrete(space.n)
        if hasattr(space, "nvec"):
            return gym.spaces.MultiDiscrete(space.nvec)
        if hasattr(space, "n") and hasattr(space, "shape"):
            return gym.spaces.MultiBinary(space.n)
        return space

    def _to_gymnasium_action_space(self, action_space):
        # SafeRL uses gym Tuple(Box, Box, ...) for multi-actuator continuous control.
        # Convert to a flat Box so gymnasium wrappers (ClipAction / vector env) can operate.
        sub_spaces = getattr(action_space, "spaces", None)
        if sub_spaces and all(hasattr(s, "low") and hasattr(s, "high") and hasattr(s, "shape") for s in sub_spaces):
            lows = []
            highs = []
            shapes = []
            for sub in sub_spaces:
                low_arr = np.asarray(sub.low, dtype=np.float32).reshape(-1)
                high_arr = np.asarray(sub.high, dtype=np.float32).reshape(-1)
                lows.append(low_arr)
                highs.append(high_arr)
                shapes.append(tuple(sub.shape))

            flat_low = np.concatenate(lows, axis=0)
            flat_high = np.concatenate(highs, axis=0)
            split_sizes = [int(np.prod(shape)) for shape in shapes]

            def unflatten(action):
                flat = np.asarray(action, dtype=np.float32).reshape(-1)
                parts = []
                start = 0
                for sz, shape in zip(split_sizes, shapes):
                    end = start + sz
                    parts.append(flat[start:end].reshape(shape))
                    start = end
                return tuple(parts)

            self._action_unflatten = unflatten
            return gym.spaces.Box(low=flat_low, high=flat_high, dtype=np.float32)

        return self._to_gymnasium_space(action_space)

    def reset(self, *, seed=None, options=None):
        if seed is not None and hasattr(self.legacy_env, "seed"):
            self.legacy_env.seed(seed)
        obs = self.legacy_env.reset()
        return obs, {}

    def step(self, action):
        legacy_action = self._action_unflatten(action) if self._action_unflatten is not None else action
        obs, reward, done, info = self.legacy_env.step(legacy_action)
        terminated, truncated = bool(done), False
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.legacy_env.render(*args, **kwargs)

    def close(self):
        return self.legacy_env.close()


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if env_id.lower() == "docking2d":
            _ensure_saferl_on_path()
            from config_aero import env_cfg
            from SafeRL.saferl.aerospace.tasks.docking.task import DockingEnv

            env = LegacyGymEnvAdapter(DockingEnv(copy.deepcopy(env_cfg)))
        else:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def evaluate(
    agent,
    make_env_fn: Callable,
    env_id: str,
    eval_episodes: int,
    obs_norm: ObsNorm,
    use_obs_norm: bool,
    run_name: str = "eval",
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    writer=None,
    global_step=0,
):
    envs = gym.vector.SyncVectorEnv([make_env_fn(env_id, 0, capture_video, run_name, 0.99)])
    agent.eval()

    obs_raw, _ = envs.reset()
    if use_obs_norm:
        obs = obs_norm.norm(obs_raw)
    else:
        obs = torch.as_tensor(obs_raw, dtype=torch.float32, device=device)
    episodic_returns = []
    episodic_lengths = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(torch.as_tensor(obs, dtype=torch.float32, device=device))
        next_obs_raw, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                episodic_returns.append(info["episode"]["r"])
                episodic_lengths.append(info["episode"]["l"])
        if use_obs_norm:
            obs = obs_norm.norm(next_obs_raw)
        else:
            obs = torch.as_tensor(next_obs_raw, dtype=torch.float32, device=device)

    ret_mean = float(np.mean(episodic_returns))
    ret_std = float(np.std(episodic_returns))
    len_mean = float(np.mean(episodic_lengths))
    if writer is not None:
        writer.add_scalar("eval/episodic_return_mean", ret_mean, global_step or 0)
        writer.add_scalar("eval/episodic_return_std", ret_std, global_step or 0)
        writer.add_scalar("eval/episodic_length_mean", len_mean, global_step or 0)
    agent.train()
    return episodic_returns


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class InputDiscretizer(nn.Module):
    def __init__(self, obs_dim: int, bits: int, wnn_xy_fov: float, wnn_vel_fov: float):
        super().__init__()
        self.bits = int(bits)
        if obs_dim == 4:
            obs_min = torch.tensor(
                [-wnn_xy_fov, -wnn_xy_fov, -wnn_vel_fov, -wnn_vel_fov],
                dtype=torch.float32,
            )
            obs_max = torch.tensor(
                [wnn_xy_fov, wnn_xy_fov, wnn_vel_fov, wnn_vel_fov],
                dtype=torch.float32,
            )
        else:
            obs_min = torch.full((obs_dim,), -wnn_xy_fov, dtype=torch.float32)
            obs_max = torch.full((obs_dim,), wnn_xy_fov, dtype=torch.float32)
        self.register_buffer("obs_min", obs_min)
        self.register_buffer("obs_max", obs_max)
        span = torch.clamp(self.obs_max - self.obs_min, min=1e-12)
        self.register_buffer("obs_span", span)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.maximum(torch.minimum(x, self.obs_max), self.obs_min)
        if self.bits <= 1:
            return self.obs_min.unsqueeze(0).expand_as(x)
        levels = float(self.bits - 1)
        scaled = (x - self.obs_min) / self.obs_span * levels
        quantized_idx = torch.round(scaled).clamp(0.0, levels)
        return self.obs_min + (quantized_idx / levels) * self.obs_span


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        float_std=0.01,
        float_size=64,
        float_layers=2,
        activation: Literal["relu", "tanh"] = "relu",
        use_discretizer: bool = False,
        bits: int = 100,
        wnn_xy_fov: float = 1.5,
        wnn_vel_fov: float = 1.0,
    ):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        hidden = int(float_size)
        num_layers = max(int(float_layers), 1)
        activation_cls = nn.ReLU if activation == "relu" else nn.Tanh
        self.input_discretizer = (
            InputDiscretizer(obs_dim, bits=bits, wnn_xy_fov=wnn_xy_fov, wnn_vel_fov=wnn_vel_fov)
            if use_discretizer
            else None
        )

        critic_layers = [layer_init(nn.Linear(obs_dim, hidden)), activation_cls()]
        for _ in range(num_layers - 1):
            critic_layers.extend([layer_init(nn.Linear(hidden, hidden)), activation_cls()])
        critic_layers.append(layer_init(nn.Linear(hidden, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        actor_layers = [layer_init(nn.Linear(obs_dim, hidden)), activation_cls()]
        for _ in range(num_layers - 1):
            actor_layers.extend([layer_init(nn.Linear(hidden, hidden)), activation_cls()])
        actor_layers.append(layer_init(nn.Linear(hidden, act_dim), std=float_std))
        self.actor_mean = nn.Sequential(*actor_layers)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        if self.input_discretizer is not None:
            x = self.input_discretizer(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.input_discretizer is not None:
            x = self.input_discretizer(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any, Sequence
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

def atanh(x, eps=1e-6):
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

class WNNActor(nn.Module):
    def __init__(self, env, args, device = 'cuda', use_tanh_final = False):
        super().__init__()

        self.args = self._normalize_args(args)
        self.device_for_init = device
        self.use_tanh_final = use_tanh_final
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))

        thermo_device = "cuda"
        # self.use_tanh_final = use_tanh_final

        thermo = ThermometerUniform(n_bits=self.args.bits, device=thermo_device)
        print("Thermometer initialized with", self.args.bits, "bits")
        if obs_dim == 4:
            min_values = torch.tensor(
                [
                    -self.args.wnn_xy_fov,
                    -self.args.wnn_xy_fov,
                    -self.args.wnn_vel_fov,
                    -self.args.wnn_vel_fov,
                ],
                device=device,
                dtype=torch.float32,
            )
            max_values = torch.tensor(
                [
                    self.args.wnn_xy_fov,
                    self.args.wnn_xy_fov,
                    self.args.wnn_vel_fov,
                    self.args.wnn_vel_fov,
                ],
                device=device,
                dtype=torch.float32,
            )
        else:
            min_values = torch.full(
                (obs_dim,), -self.args.wnn_xy_fov, device=device, dtype=torch.float32
            )
            max_values = torch.full(
                (obs_dim,), self.args.wnn_xy_fov, device=device, dtype=torch.float32
            )

        thermo.fit(
            torch.zeros((1, obs_dim), device=device),
            min_value=min_values,
            max_value=max_values,
        )

        # if int(self.args.bits) <= 1:
        #     boundary_thresholds = min_values.unsqueeze(1)
        # #Q: Why is this necessary? Shouldn't the thermometer just handle this case internally?
        # else:
        #     frac = (
        #         torch.arange(int(self.args.bits), device=device, dtype=torch.float32)
        #         / float(int(self.args.bits) - 1)
        #     )
        #     boundary_thresholds = (
        #         min_values.unsqueeze(1)
        #         + frac.unsqueeze(0) * (max_values - min_values).unsqueeze(1)
        #     )

        # thermo.thresholds = boundary_thresholds

        self.actor_mean = WNN(
            obs_dim=obs_dim,
            act_dim=act_dim,
            sizes=[self.args.size] * self.args.l,
            thermometer=thermo,
            bits=self.args.bits,
            n=self.args.n,
            init_log_alpha=self.args.init_log_alpha,
            later_learnable=False,
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        self.register_buffer("obs_min", min_values)
        self.register_buffer("obs_max", max_values)
        self._z3_encoder_cache = None
        self._z3_encoder_cache_affine = None

    @staticmethod
    def _normalize_args(args) -> SimpleNamespace:
        """
        Accepts argparse.Namespace, dataclass instance, SimpleNamespace, or dict,
        and returns a SimpleNamespace.
        """
        if isinstance(args, dict):
            return SimpleNamespace(**copy.deepcopy(args))
        if is_dataclass(args):
            return SimpleNamespace(**asdict(args))
        if hasattr(args, "__dict__"):
            return SimpleNamespace(**copy.deepcopy(vars(args)))
        raise TypeError(f"Unsupported args type: {type(args)}")

    def get_config(self) -> dict[str, Any]:
        return copy.deepcopy(vars(self.args))

    def _clip_obs(self, x):
        return torch.maximum(torch.minimum(x, self.obs_max), self.obs_min)

    def get_value(self, x):
        x_clipped = self._clip_obs(x)
        return self.critic(x_clipped)

    def get_action_and_value(self, x, action=None, deterministic=False):
        x_clipped = self._clip_obs(x)
        action_mean = self.actor_mean(x_clipped)
        
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)#
        if self.use_tanh_final:
            # action returned to the environment is bounded in [-1, 1]
            # latent_action is the pre-tanh Gaussian sample
            if action is None:
                if deterministic:
                    latent_action = action_mean
                else:
                    # rsample is usually preferred for differentiable sampling,
                    # though PPO does not strictly require it.
                    latent_action = probs.rsample()

                action = torch.tanh(latent_action)
            else:
                # If an already-squashed action is passed in, invert tanh
                # to recover the latent action needed for Normal.log_prob.
                latent_action = atanh(action)

            # Gaussian log-prob in latent space
            log_prob = probs.log_prob(latent_action)

            # Change-of-variables correction:
            # log p(a) = log p(z) - log |det da/dz|
            # where a = tanh(z), da/dz = 1 - tanh(z)^2
            log_prob -= torch.log(1 - torch.tanh(latent_action).pow(2) + 1e-6)

            log_prob = log_prob.sum(1)

            # Entropy of tanh-squashed Gaussian has no simple closed form.
            # This is the entropy of the unsquashed Gaussian, commonly used
            # as an approximation in PPO.
            entropy = probs.entropy().sum(1)
        else:
            if action is None:
                if deterministic:
                    action = action_mean
                else:
                    action = probs.sample()

            log_prob = probs.log_prob(action).sum(1)
            entropy = probs.entropy().sum(1)

        value = self.critic(x_clipped)

        return action, log_prob, entropy, value
        
    def forward(self, x, deterministic=False):
        return self.get_action_and_value(x, deterministic=deterministic)[0]

    def _build_z3_payload(self) -> dict[str, Any]:
        actor_sd = {f"actor_mean.{k}": v.detach().cpu() for k, v in self.actor_mean.state_dict().items()}
        actor_sd["actor_logstd"] = self.actor_logstd.detach().cpu()
        input_ranges = [
            (float(lo), float(hi))
            for lo, hi in zip(
                self.obs_min.detach().cpu().tolist(),
                self.obs_max.detach().cpu().tolist(),
            )
        ]
        print("Input ranges:", self.actor_mean.thermometer.thresholds.detach().cpu() )
        return {
            "state_dict": actor_sd,
            "input_ranges": input_ranges,
            "thermometer_type": "uniform",
            "thermo_thresholds": self.actor_mean.thermometer.thresholds.detach().cpu() if self.actor_mean.thermometer is not None else None,
            "uses_obs_norm": False,
            "config": self.get_config(),
        }

    def get_z3_encoder(self, include_affine: bool = True, refresh: bool = False):
        if (
            (not refresh)
            and self._z3_encoder_cache is not None
            and self._z3_encoder_cache_affine == bool(include_affine)
        ):
            return self._z3_encoder_cache
        from wnn_z3_encoder import WNNZ3Encoder

        payload = self._build_z3_payload()
        #print(payload.keys())
        #exit()
        enc = WNNZ3Encoder.from_state_dict(
            payload,
            include_affine=bool(include_affine),
            actor_prefix="actor_mean",
            little_endian_lut_index=True,
        )
        self._z3_encoder_cache = enc
        self._z3_encoder_cache_affine = bool(include_affine)
        return enc

    def z3_expr(
        self,
        x_exprs: Sequence[object],
        include_affine: bool = True,
        include_obs_norm: bool = False,
        inputs_are_bits: bool = False,
    ) -> list[object]:
        enc = self.get_z3_encoder(include_affine=include_affine)
        return enc.encode_z3(
            x_exprs,
            include_affine=include_affine,
            include_obs_norm=include_obs_norm,
            inputs_are_bits=inputs_are_bits,
        )

    def save_checkpoint(self, path, optimizer=None, extra = None):
        payload = {
            "model_state_dict": self.state_dict(),
            "args": self.get_config(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra is not None:
            payload["extra"] = extra

        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, env, path, device = 'cuda', strict: bool = True):
        checkpoint = torch.load(path, map_location=device)

        if "args" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain saved args. "
                "Save checkpoints with an 'args' field."
            )

        args = checkpoint["args"]
        print(args)
        model = cls(env=env, args=args, device=device)
        print("-------------")
        print(checkpoint["model_state_dict"]['actor_mean.net.1.luts'].shape)
        print(checkpoint["model_state_dict"]['actor_mean.net.2.luts'].shape)
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        model.to(device)
        return model

    def load_controller_to_env(self, env, path, device: torch.device = None):
        """
        If you want to mutate an existing instance instead of constructing a new one.
        Usually from_checkpoint(...) is cleaner.
        """
        if device is None:
            device = next(self.parameters()).device
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        
    def ctrl_env_step(self, x, env):
        action = self(x, deterministic=True)
        return env(x, action), action

    def freeze_interconnect(self):
        """Freeze learnable mapping/interconnect weights in LUT layers."""
        frozen_names = []
        for name, param in self.actor_mean.named_parameters():
            if name.endswith("mapping.weights"):
                param.requires_grad_(False)
                param.grad = None
                frozen_names.append(name)
        return frozen_names
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    args = apply_preset(args)

    if args.network_type == "lgn":
        args.network_type = "wnn"
    if not (0.0 <= args.freeze_wnn_interconnect_after_frac <= 1.0):
        raise ValueError(
            f"freeze_wnn_interconnect_after_frac must be in [0, 1], got {args.freeze_wnn_interconnect_after_frac}"
        )

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    #args.eval_every_iterations = args.num_iterations // 10 if args.eval_every_iterations <= 0 else args.eval_every_iterations
    #print(args.total_timesteps)
    #exit()
    # print(args)

    run_name = f"{args.env_id}__{args.exp_name}_{args.network_type}__{args.seed}__{int(time.time())}"
    if args.track:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
                group=f"{args.bits}_{args.size}_{args.env_id}_{args.exp_name}_{args.network_type}",
            )
        except Exception as exc:
            print(f"[WARN] W&B tracking requested but unavailable: {exc}")
            print("[WARN] Continuing without W&B tracking.")
            args.track = False
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    if args.network_type == "wnn":
        if device.type != "cuda":
            raise RuntimeError("WNN/LGN currently requires CUDA because torch_dwn CPU kernels are not implemented.")
        agent = WNNActor(envs, args, device=device, use_tanh_final=args.use_tanh_final).to(device)
    elif args.network_type in {"relu", "tanh"}:
        agent = Agent(
            envs,
            args.float_std,
            float_size=args.float_size,
            float_layers=args.float_layers,
            activation=args.network_type,
            use_discretizer=args.use_discretizer,
            bits=args.bits,
            wnn_xy_fov=args.wnn_xy_fov,
            wnn_vel_fov=args.wnn_vel_fov,
        ).to(device)
        if args.use_discretizer:
            print(
                f"Enabled float input discretizer: bits={args.bits}, "
                f"xy_fov={args.wnn_xy_fov}, vel_fov={args.wnn_vel_fov}"
            )
    else:
        raise NotImplementedError(f"Unsupported network_type={args.network_type}")
    use_obs_norm = bool(args.use_obs_norm)
    if args.network_type == "wnn" and use_obs_norm:
        print("[WARN] use_obs_norm=True ignored for WNN (WNN uses raw clipped observations).")
        use_obs_norm = False

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    freeze_step = int(args.total_timesteps * args.freeze_wnn_interconnect_after_frac)
    interconnect_frozen = False

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    obs_shape = tuple(envs.single_observation_space.shape)
    obs_norm = ObsNorm(obs_shape, device=device)
    if use_obs_norm and args.obs_norm_load_path:
        state = torch.load(args.obs_norm_load_path, map_location=device)
        obs_norm.load_state_dict(state)
        print(f"Loaded ObsNorm from {args.obs_norm_load_path}")
    elif (not use_obs_norm) and args.obs_norm_load_path:
        print(f"[WARN] Ignoring --obs-norm-load-path because use_obs_norm={use_obs_norm}.")

    next_obs_raw, _ = envs.reset(seed=args.seed)
    if use_obs_norm:
        obs_norm.update(next_obs_raw)
        next_obs = obs_norm.norm(next_obs_raw)
    else:
        next_obs = torch.as_tensor(next_obs_raw, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    print(args.num_iterations)
    print(args.num_steps)
    print(args.num_envs)
    #print(args.epochs)
    print('Total global steps per iteration:', args.num_steps * args.num_envs * args.num_iterations)
    # exit()
    for iteration in tqdm(range(1, args.num_iterations + 1)):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_raw, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, device=device).view(-1)
            if use_obs_norm:
                obs_norm.update(next_obs_raw)
                next_obs = obs_norm.norm(next_obs_raw)
            else:
                next_obs = torch.as_tensor(next_obs_raw, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        if (
            args.network_type == "wnn"
            and args.freeze_wnn_interconnect
            and (not interconnect_frozen)
            and global_step >= freeze_step
        ):
            frozen_names = agent.freeze_interconnect()
            interconnect_frozen = True
            if frozen_names:
                print(
                    f"Froze WNN interconnect mapping at global_step={global_step} "
                    f"(threshold={freeze_step}): {frozen_names}"
                )
                writer.add_scalar("wnn/interconnect_frozen", 1.0, global_step)
            else:
                print(
                    f"[WARN] freeze_wnn_interconnect enabled but no learnable mapping parameters were found "
                    f"at global_step={global_step}."
                )
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

        eval_every = args.eval_every_iterations if args.eval_every_iterations > 0 else max(args.num_iterations // 20, 10)
        if (iteration - 1) % eval_every == 0:
            returns = evaluate(
                agent,
                make_env,
                args.env_id,
                eval_episodes=args.eval_episodes,
                obs_norm=obs_norm,
                use_obs_norm=use_obs_norm,
                device=device,
                capture_video=False,
                writer=writer,
                global_step=global_step,
            )
            ret_mean = float(np.mean(returns))
            ret_std = float(np.std(returns))
            print(f"Evaluation at iteration {iteration}: mean_return={ret_mean:.2f}, std_return={ret_std:.2f}")

    final_returns = evaluate(
        agent,
        make_env,
        args.env_id,
        eval_episodes=args.final_eval_episodes,
        obs_norm=obs_norm,
        use_obs_norm=use_obs_norm,
        device=device,
        capture_video=False,
        writer=writer,
        global_step=global_step + 1,
    )

    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    if use_obs_norm:
        obs_norm_path = os.path.join(run_dir, args.obs_norm_save_name)
        torch.save(obs_norm.state_dict(), obs_norm_path)
        print(f"ObsNorm saved to {obs_norm_path}")
    reward_mean = float(np.mean(final_returns)) if len(final_returns) > 0 else float("nan")
    reward_std = float(np.std(final_returns)) if len(final_returns) > 0 else float("nan")
    reward_path = os.path.join(run_dir, "reward_achieved")
    with open(reward_path, "w") as f:
        f.write(f"mean_reward={reward_mean}\n")
        f.write(f"std_reward={reward_std}\n")
        f.write(f"episodes={len(final_returns)}\n")
    print(f"Final reward written to {reward_path}")

    print(run_name)
    save_dir = os.path.join(args.save_path, run_name)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"BeforeCEGIS_{args.exp_name}.cleanrl_model")
    agent.save_checkpoint(model_path, optimizer=optimizer)
            
    #_, certificate = train_controller_and_certificate(agent)
    _, certificate = verification_loop(agent)

    if args.save_model:
        print(run_name)
        save_dir = os.path.join(args.save_path, run_name)
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{args.exp_name}.cleanrl_model")
        certificate_path = os.path.join(save_dir, f"{args.exp_name}.z3_certificate")
        if args.network_type == "wnn":
            agent.save_checkpoint(model_path, optimizer=optimizer)
            if certificate is not None:
                torch.save(certificate, certificate_path)
        else:
            torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        if use_obs_norm:
            torch.save(obs_norm.state_dict(), os.path.join(save_dir, f"{args.obs_norm_save_name}"))
        with open(os.path.join(save_dir, "reward_achieved"), "w") as f:
            f.write(f"mean_reward={reward_mean}\n")
            f.write(f"std_reward={reward_std}\n")
            f.write(f"episodes={len(final_returns)}\n")

    envs.close()
    writer.close()
