#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
import numpy as np
import torch

from ppo_train_docking import Agent, ObsNorm, WNNActor, make_env

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved PPO docking controller and report safety/success/reward statistics."
        )
    )
    parser.add_argument(
        "--controller",
        type=str,
        required=True,
        help=(
            "Either a run folder name under --save-path (e.g. "
            "Docking2d__ppo_train_docking_wnn__1__1775228448) or a direct path "
            "to a .cleanrl_model/.onnx file or containing folder."
        ),
    )
    parser.add_argument("--save-path", type=str, default="models_ppo")
    parser.add_argument(
        "--model-file",
        type=str,
        default="",
        help="Optional explicit model filename inside the controller folder.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-id", type=str, default="Docking2d")
    parser.add_argument(
        "--env-config-file",
        type=str,
        default="config_aero.py",
        help=(
            "Docking environment config module/file to load `env_cfg` from "
            "(e.g. config_aero.py or config_aero_default.py)."
        ),
    )
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no-cuda", action="store_false", dest="cuda")
    parser.add_argument(
        "--network-type",
        type=str,
        choices=["auto", "wnn", "float", "relu", "tanh", "lgn", "onnx"],
        default="wnn",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (actor mean).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_false",
        dest="deterministic",
        help="Sample from the policy distribution during evaluation.",
    )
    parser.add_argument(
        "--use-obs-norm",
        action="store_true",
        default=False,
        help="Apply observation normalization if obs stats are available.",
    )
    parser.add_argument(
        "--obs-norm-path",
        type=str,
        default="",
        help="Optional explicit path to obs_norm.pt.",
    )
    parser.add_argument("--float-size", type=int, default=64)
    parser.add_argument("--float-layers", type=int, default=2)
    parser.add_argument("--float-std", type=float, default=0.01)
    parser.add_argument(
        "--safety-n",
        type=float,
        default=0.001027,
        help=(
            "Mean-motion n for safety check. If omitted, inferred from the DockingEnv dynamics "
            "(fallback: 0.001027)."
        ),
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="eval_trajectories.png",
        help="Output path for trajectory plot.",
    )
    parser.add_argument(
        "--velocity-plot-path",
        type=str,
        default="eval_velocity_magnitude.png",
        help="Output path for velocity-magnitude-vs-timestep plot.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate a trajectory plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Disable trajectory plotting.",
    )
    return parser.parse_args()
from torch import nn
class LearnedController(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4,4,bias=False),
            nn.Linear(4,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,4),
            nn.Linear(4,2,bias=False),
        )
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def actor_mean(self, x):
        return self.forward(x)


class ONNXController:
    def __init__(self, model_path: Path, device: torch.device):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX evaluation requires onnxruntime. Install onnxruntime "
                "or onnxruntime-gpu in this environment."
            ) from exc

        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            available = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def eval(self):
        return self

    def actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
        out = self.session.run([self.output_name], {self.input_name: x_np})[0]
        return torch.as_tensor(out, dtype=torch.float32, device=x.device)
    
def resolve_model_path(controller: str, save_path: str, model_file: str):
    controller_path = Path(controller)
    if controller_path.is_file():
        return controller_path.resolve(), controller_path.parent.resolve()

    if controller_path.is_dir():
        run_dir = controller_path.resolve()
    else:
        run_dir = (Path(save_path) / controller).resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Controller directory not found: {run_dir}")

    if model_file:
        model_path = (run_dir / model_file).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path, run_dir

    preferred = run_dir / "ppo_train_docking.cleanrl_model"
    if preferred.is_file():
        return preferred.resolve(), run_dir

    preferred_onnx = run_dir / "model.onnx"
    if preferred_onnx.is_file():
        return preferred_onnx.resolve(), run_dir

    candidates = sorted(run_dir.glob("*.cleanrl_model")) + sorted(run_dir.glob("*.onnx"))
    if len(candidates) == 1:
        return candidates[0].resolve(), run_dir
    if len(candidates) == 0:
        raise FileNotFoundError(f"No .cleanrl_model or .onnx model found in {run_dir}")
    raise RuntimeError(
        "Multiple model files found (.cleanrl_model/.onnx). Pass --model-file explicitly. "
        f"Candidates: {[p.name for p in candidates]}"
    )


def infer_network_type(requested: str, controller: str, model_path: Path):
    if model_path.suffix.lower() == ".onnx":
        return "onnx"
    if requested == "lgn":
        return "wnn"
    if requested == "float":
        # Legacy alias: historical float network used ReLU activations.
        return "relu"
    if requested != "auto":
        return requested

    controller_lower = controller.lower()
    if "__wnn__" in controller_lower or "__lgn__" in controller_lower:
        return "wnn"
    if "__relu__" in controller_lower:
        return "relu"
    if "__tanh__" in controller_lower:
        return "tanh"
    if "__float__" in controller_lower:
        return "relu"

    payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, dict):
        if "args" in payload and isinstance(payload["args"], dict):
            net = str(payload["args"].get("network_type", "wnn")).lower()
            if net == "lgn":
                return "wnn"
            if net == "float":
                return "relu"
            if net in {"relu", "tanh", "wnn"}:
                return net
            return "wnn"
        if "model_state_dict" in payload:
            return "wnn"
    return "relu"


def resolve_obs_norm_path(run_dir: Path, obs_norm_path: str) -> Optional[Path]:
    if obs_norm_path:
        p = Path(obs_norm_path)
        if p.is_file():
            return p.resolve()
        p_run = (run_dir / obs_norm_path).resolve()
        if p_run.is_file():
            return p_run
        raise FileNotFoundError(f"Obs norm file not found: {obs_norm_path}")

    p_default = (run_dir / "obs_norm.pt").resolve()
    if p_default.is_file():
        return p_default
    return None


def build_agent(model_path: Path, network_type: str, args, device, eval_env, load_from_p=True):
    env_view = SimpleNamespace(
        single_observation_space=eval_env.observation_space,
        single_action_space=eval_env.action_space,
    )
    if network_type == "onnx":
        return ONNXController(model_path=model_path, device=device)

    if network_type == "wnn":
        if device.type != "cuda":
            raise RuntimeError(
                "WNN checkpoints require CUDA in this codebase "
                "(torch_dwn CPU kernels are not implemented)."
            )
        agent = WNNActor.from_checkpoint(env=env_view, path=str(model_path), device=device)
        agent.use_tanh_final = True
        return agent.to(device)

    if network_type in {"float", "relu", "tanh"} and not load_from_p:
        activation = "tanh" if network_type == "tanh" else "relu"
        agent = Agent(
            env_view,
            float_std=args.float_std,
            float_size=args.float_size,
            float_layers=args.float_layers,
            activation=activation,
        ).to(device)
        state = torch.load(model_path, map_location=device)
        #print(state.keys())
        #exit()
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        agent.load_state_dict(state, strict=True)
        return agent
    elif load_from_p:
        agent = LearnedController().to(device)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        agent.load_state_dict(state, strict=True)
        print("Loaded model state dict keys: ", state.keys())
        return agent

    raise ValueError(f"Unsupported network type: {network_type}")


def infer_safety_n(eval_env):
    cur = eval_env
    for _ in range(16):
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    legacy_env = getattr(cur, "legacy_env", None)
    if legacy_env is None:
        return None
    env_objs = getattr(legacy_env, "env_objs", None)
    if not isinstance(env_objs, dict):
        return None
    for name in ("deputy", "chief"):
        obj = env_objs.get(name)
        n_val = getattr(getattr(obj, "dynamics", None), "n", None)
        if n_val is not None:
            return float(n_val)
    return None


def is_safe_step(obs, safety_n):
    x0 = np.asarray(obs, dtype=np.float64).reshape(-1)
    if x0.size < 4:
        return True
    # multiply x and y with 100 and divide velocity by 2. Do not change x0
    x = x0.copy()
    x[0] = x[0] * 100.0
    x[1] = x[1] * 100.0
    x[2] = x[2] / 2.0
    x[3] = x[3] / 2.0
    
    pos_norm = np.sqrt(x[0] * x[0] + x[1] * x[1])
    rhs = 0.2 + (2.0 * float(safety_n)) * pos_norm
    vel_euc = np.sqrt(x[2] * x[2] + x[3] * x[3])
    unsafe_m = vel_euc > rhs
   # print('vel_euc: ', vel_euc, 'rhs: ', rhs, 'safe: ', not(unsafe_m), obs)
    # print('is safe: ', bool(unsafe_m))
    # print the velocity and the rhs and all the info if unsafe
    #if bool(unsafe_m):
        #print('..........')
        #print('velocity squared: ', vel_euc, rhs)
        #print('observations: ', x0, 'pos norm: ', pos_norm, 'rhs: ', rhs)
    return not bool(unsafe_m)


def get_action(agent, obs_tensor, deterministic, network_type):
    with torch.no_grad():
        if deterministic:
            if network_type == "wnn":
                action = agent(obs_tensor, deterministic=True)
            else:
                action = agent.actor_mean(obs_tensor)
        else:
            if network_type == "onnx":
                raise ValueError(
                    "Stochastic evaluation is not supported for ONNX models. "
                    "Use --deterministic."
                )
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
    return action.squeeze(0).detach().cpu().numpy()


def obs_xy(obs):
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return float("nan"), float("nan")
    return float(arr[0]), float(arr[1])


def infer_velocity_denorm_mapping(eval_env):
    """
    Infer which observation entries correspond to velocity components and the
    normalization constants used by SafeRL observation processors.
    Returns list of (obs_index, sigma) where physical_value = observed_value * sigma.
    """
    cur = eval_env
    for _ in range(16):
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    legacy_env = getattr(cur, "legacy_env", None)
    if legacy_env is None:
        return [(2, 1.0), (3, 1.0)]

    obs_manager = getattr(legacy_env, "observation_manager", None)
    processors = getattr(obs_manager, "processors", None)
    if not isinstance(processors, list):
        return [(2, 1.0), (3, 1.0)]

    def sigma_at(norm, local_idx):
        if norm is None:
            return 1.0
        arr = np.asarray(norm, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return 1.0
        if arr.size == 1:
            return float(arr[0])
        if local_idx < arr.size:
            return float(arr[local_idx])
        return 1.0

    mapping = []
    offset = 0
    for proc in processors:
        shape = getattr(getattr(proc, "observation_space", None), "shape", None)
        width = int(np.prod(shape)) if shape is not None else 1
        norm = getattr(proc, "normalization", None)
        cls_name = type(proc).__name__

        if hasattr(proc, "attr"):
            attr = str(getattr(proc, "attr", "")).lower()
            if attr in {"x_dot", "y_dot"}:
                mapping.append((offset, sigma_at(norm, 0)))
        elif cls_name == "DockingObservationProcessor":
            mode = str(getattr(proc, "mode", "2d")).lower()
            if mode == "2d" and width >= 4:
                mapping.append((offset + 2, sigma_at(norm, 2)))
                mapping.append((offset + 3, sigma_at(norm, 3)))
            elif mode == "3d" and width >= 6:
                mapping.append((offset + 3, sigma_at(norm, 3)))
                mapping.append((offset + 4, sigma_at(norm, 4)))
                mapping.append((offset + 5, sigma_at(norm, 5)))

        offset += width

    if len(mapping) < 2:
        return [(2, 1.0), (3, 1.0)]
    return mapping


def infer_obs_denorm_vector(eval_env, obs_dim: int):
    """
    Infer per-dimension sigma used by SafeRL observation processors.
    Returns vector `sigma` so that physical_obs ~= observed_obs * sigma.
    """
    cur = eval_env
    for _ in range(16):
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    legacy_env = getattr(cur, "legacy_env", None)
    if legacy_env is None:
        return np.ones(obs_dim, dtype=np.float64)

    obs_manager = getattr(legacy_env, "observation_manager", None)
    processors = getattr(obs_manager, "processors", None)
    if not isinstance(processors, list):
        return np.ones(obs_dim, dtype=np.float64)

    sigmas = []
    for proc in processors:
        shape = getattr(getattr(proc, "observation_space", None), "shape", None)
        width = int(np.prod(shape)) if shape is not None else 1
        norm = getattr(proc, "normalization", None)
        if norm is None:
            sigmas.extend([1.0] * width)
            continue

        arr = np.asarray(norm, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            sigmas.extend([1.0] * width)
        elif arr.size == 1:
            sigmas.extend([float(arr[0])] * width)
        elif arr.size >= width:
            sigmas.extend([float(v) for v in arr[:width]])
        else:
            sigmas.extend([float(v) for v in arr])
            sigmas.extend([1.0] * (width - arr.size))

    if len(sigmas) < obs_dim:
        sigmas.extend([1.0] * (obs_dim - len(sigmas)))
    return np.asarray(sigmas[:obs_dim], dtype=np.float64)


def obs_velocity_magnitude(obs, velocity_denorm_mapping):
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    components = []
    for obs_idx, sigma in velocity_denorm_mapping:
        if obs_idx < 0 or obs_idx >= arr.size:
            continue
        components.append(float(arr[obs_idx]) * float(sigma))
    if len(components) == 0:
        return float("nan")
    return float(np.sqrt(np.sum(np.square(components))))


def plot_trajectories(trajectories, plot_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    failure_labeled = False
    success_labeled = False
    violation_labeled = False
    state_labeled = False

    for traj in trajectories:
        xy = np.asarray(traj["xy"], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
            continue

        if traj["failed"]:
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="#d95f02",
                alpha=0.65,
                linewidth=1.6,
                label="Failed trajectory" if not failure_labeled else None,
            )
            failure_labeled = True
        else:
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="#1b9e77",
                alpha=0.30,
                linewidth=1.0,
                label="Successful trajectory" if not success_labeled else None,
            )
            success_labeled = True

        safety_violation_mask = np.asarray(traj.get("safety_violation_mask", []), dtype=bool).reshape(-1)
        if safety_violation_mask.size != xy.shape[0]:
            safety_violation_mask = np.zeros((xy.shape[0],), dtype=bool)

        safe_xy = xy[~safety_violation_mask]
        if safe_xy.size > 0:
            ax.scatter(
                safe_xy[:, 0],
                safe_xy[:, 1],
                s=12,
                marker="o",
                c="#4c4c4c",
                alpha=0.55,
                linewidths=0.0,
                label="State (safe)" if not state_labeled else None,
            )
            state_labeled = True

        viol_xy = xy[safety_violation_mask]
        if viol_xy.size > 0:
            ax.scatter(
                viol_xy[:, 0],
                viol_xy[:, 1],
                s=28,
                marker="x",
                c="#e7298a",
                linewidths=1.1,
                label="State (safety violation)" if not violation_labeled else None,
            )
            violation_labeled = True

    ax.scatter([0.0], [0.0], c="black", s=35, marker="o", label="Docking target")
    ax.set_title("Docking Evaluation Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = Path(plot_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_velocity_magnitude_over_time(trajectories, plot_path):
    vel_trajs = [np.asarray(t.get("velocity_magnitude", []), dtype=np.float64) for t in trajectories]
    vel_trajs = [v for v in vel_trajs if v.size > 0]
    if len(vel_trajs) == 0:
        return None

    max_len = max(v.shape[0] for v in vel_trajs)
    data = np.full((len(vel_trajs), max_len), np.nan, dtype=np.float64)
    for i, v in enumerate(vel_trajs):
        data[i, : v.shape[0]] = v

    counts = np.sum(~np.isnan(data), axis=0)
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    se = np.divide(std, np.sqrt(np.maximum(counts, 1)), out=np.zeros_like(std), where=counts > 0)
    ci95 = 1.96 * se

    timesteps = np.arange(max_len, dtype=np.int64)
    lower = mean - ci95
    upper = mean + ci95

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(timesteps, mean, color="#1f77b4", linewidth=2.0, label="Mean velocity magnitude")
    ax.fill_between(
        timesteps,
        lower,
        upper,
        color="#1f77b4",
        alpha=0.22,
        label="95% confidence band",
    )
    ax.set_title("Velocity Magnitude vs Timestep")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Velocity magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = Path(plot_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_path, run_dir = resolve_model_path(args.controller, args.save_path, args.model_file)
    network_type = infer_network_type(args.network_type, args.controller, model_path)
    print(network_type)
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

    if network_type == "wnn" and device.type != "cuda":
        raise RuntimeError(
            "WNN evaluation needs CUDA. Re-run with GPU available or evaluate a float checkpoint."
        )

    eval_env = make_env(
        args.env_id,
        idx=0,
        capture_video=False,
        run_name="eval",
        gamma=0.99,
        env_config_file=args.env_config_file,
    )()
    velocity_denorm_mapping = infer_velocity_denorm_mapping(eval_env)
    obs_dim = int(np.prod(eval_env.observation_space.shape))
    obs_denorm_sigma = infer_obs_denorm_vector(eval_env, obs_dim)
    agent = build_agent(model_path, network_type, args, device, eval_env, load_from_p=False)
    agent.eval()
    print('using tanh final:', agent.use_tanh_final)
    # exit()
    if args.safety_n is not None:
        safety_n = float(args.safety_n)
    else:
        safety_n = infer_safety_n(eval_env)
        if safety_n is None:
            safety_n = 0.001027
            print("[WARN] Could not infer safety_n from environment; using fallback safety_n=0.001027")

    obs_norm = ObsNorm(tuple(eval_env.observation_space.shape), device=device)
    obs_norm_path = resolve_obs_norm_path(run_dir, args.obs_norm_path)
    use_obs_norm = bool(args.use_obs_norm)
    if network_type == "wnn" and use_obs_norm:
        print("[WARN] --use-obs-norm ignored for WNN checkpoints.")
        use_obs_norm = False
    if network_type == "onnx" and use_obs_norm:
        print("[WARN] --use-obs-norm ignored for ONNX models (ONNX expects unnormalized inputs).")
        use_obs_norm = False
    if obs_norm_path is not None:
        obs_norm.load_state_dict(torch.load(obs_norm_path, map_location=device))
        if use_obs_norm:
            print(f"Loaded obs norm from {obs_norm_path}")
    elif use_obs_norm:
        print("[WARN] --use-obs-norm requested but no obs_norm.pt found; running without obs norm.")
        use_obs_norm = False

    episode_rewards = []
    episode_successes = []
    strict_successes = []
    episode_safe = []
    success_steps = []
    trajectories = []
    from tqdm import tqdm
    for ep in tqdm(range(args.episodes)):
        obs_raw, _ = eval_env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_is_safe = True
        final_info = {}
        xy_points = []
        safety_violation_xy = []
        safety_violation_mask = []
        velocity_magnitude = []
        entered_docking = False
        first_docking_step = None

        x0, y0 = obs_xy(obs_raw)
        xy_points.append([x0, y0])
        init_step_safe = is_safe_step(obs_raw, safety_n)
        safety_violation_mask.append(not init_step_safe)
        if not init_step_safe:
            safety_violation_xy.append([x0, y0])
        ep_is_safe = ep_is_safe and init_step_safe
        velocity_magnitude.append(obs_velocity_magnitude(obs_raw, velocity_denorm_mapping))

        while not done:
            obs_for_policy = np.asarray(obs_raw, dtype=np.float64).reshape(-1)
            if network_type == "onnx":
                obs_for_policy = obs_for_policy * obs_denorm_sigma
            obs_tensor = torch.as_tensor(obs_for_policy, dtype=torch.float32, device=device).unsqueeze(0)
            if use_obs_norm and network_type != "onnx":
                obs_tensor = obs_norm.norm(obs_tensor)
            action = get_action(agent, obs_tensor, args.deterministic, network_type)
            obs_raw, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            step_safe = is_safe_step(obs_raw, safety_n)
            # print(step_safe)
            ep_is_safe = ep_is_safe and step_safe
            status = info.get("status", {})
            if isinstance(status, dict) and bool(status.get("in_docking", False)):
                entered_docking = True
                if first_docking_step is None:
                    first_docking_step = ep_steps
            x, y = obs_xy(obs_raw)
            xy_points.append([x, y])
            safety_violation_mask.append(not step_safe)
            velocity_magnitude.append(obs_velocity_magnitude(obs_raw, velocity_denorm_mapping))
            if not step_safe:
                safety_violation_xy.append([x, y])
            done = bool(terminated or truncated)
            final_info = info
            #print(reward)
        #exit()
        strict_success = bool(final_info.get("success", False))
        if not strict_success and isinstance(final_info.get("status"), dict):
            strict_success = bool(final_info["status"].get("success", False))

        # Custom eval definition:
        # reaching docking region counts as docking success even if safety is violated.
        success = bool(entered_docking or strict_success)
        failed = not success

        episode_rewards.append(ep_reward)
        episode_successes.append(success)
        strict_successes.append(strict_success)
        episode_safe.append(ep_is_safe)
        if success:
            if first_docking_step is not None:
                success_steps.append(first_docking_step)
            else:
                success_steps.append(ep_steps)
        trajectories.append(
            {
                "xy": xy_points,
                "safety_violation_xy": safety_violation_xy,
                "safety_violation_mask": safety_violation_mask,
                "velocity_magnitude": velocity_magnitude,
                "failed": failed,
            }
        )

    eval_env.close()

    rewards = np.asarray(episode_rewards, dtype=np.float64)
    successes = np.asarray(episode_successes, dtype=np.float64)
    strict_succ = np.asarray(strict_successes, dtype=np.float64)
    safe_mask = np.asarray(episode_safe, dtype=np.float64)

    success_rate = float(successes.mean()) if len(successes) else float("nan")
    strict_success_rate = float(strict_succ.mean()) if len(strict_succ) else float("nan")
    safety_success = float(safe_mask.mean()) if len(safe_mask) else float("nan")
    violation_rate = 1.0 - safety_success if np.isfinite(safety_success) else float("nan")
    reward_mean = float(rewards.mean()) if len(rewards) else float("nan")
    reward_std = float(rewards.std()) if len(rewards) else float("nan")
    avg_steps_to_docking = float(np.mean(success_steps)) if len(success_steps) else float("nan")

    print("=" * 72)
    print("Docking PPO Evaluation")
    print("=" * 72)
    print(f"controller: {args.controller}")
    print(f"model_path: {model_path}")
    print(f"network_type: {network_type}")
    print(f"episodes: {args.episodes}")
    print(f"deterministic: {args.deterministic}")
    print(f"device: {device}")
    print(f"safety_n: {float(safety_n):.8f}")
    print("-" * 72)
    print(
        f"Safety success (trajectory has NO safety violation): "
        f"{safety_success:.4f} ({safety_success * 100:.2f}%)"
    )
    print(f"Safety violation rate (inverse): {violation_rate:.4f} ({violation_rate * 100:.2f}%)")
    print(
        f"Docking success rate (entered docking region): "
        f"{success_rate:.4f} ({success_rate * 100:.2f}%)"
    )
    print(
        f"Strict success rate (entered region + velocity constraint): "
        f"{strict_success_rate:.4f} ({strict_success_rate * 100:.2f}%)"
    )
    print(f"Average steps till docking (docking-success episodes only): {avg_steps_to_docking:.2f}")
    print(f"Average reward: {reward_mean:.6f}")
    print(f"Reward standard deviation: {reward_std:.6f}")
    print("=" * 72)
    if args.plot:
        plot_file = plot_trajectories(trajectories, args.plot_path)
        print(f"Trajectory plot saved to: {plot_file}")
        vel_plot_file = plot_velocity_magnitude_over_time(trajectories, args.velocity_plot_path)
        if vel_plot_file is not None:
            print(f"Velocity-magnitude plot saved to: {vel_plot_file}")
        else:
            print("[WARN] Velocity-magnitude plot skipped (no valid velocity data).")


if __name__ == "__main__":
    main()
