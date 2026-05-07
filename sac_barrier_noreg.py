# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from typing import Optional, Literal
from torch.utils.tensorboard import SummaryWriter

from utils.buffers import ReplayBuffer
from wnn_models import WNN

from quantized_models import QuantizedModel 
from brevitas.nn import QuantReLU
from barrier import CTRL_INTERVAL, CtrlRoomTemp, TEMP_DISC, CTRL_DISC, transition, MIN_TEMP, MAX_TEMP

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "sac_WNN_1311"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""

    ## GENERAL ##
    network_type: Literal["wnn", "float", "quant"] = "wnn"
    """Either wnn, float or quant"""
    l : int = 2
    """Number of Layers"""
    running_normalization: bool = False
    """Running Normalization"""
    save_path: str = "models"
    """The directory to save the final model to"""
    final_evaluation_episodes : int = 1000
    
    # WNN         
    size: int = 512
    """the size of the input for the WNN"""
    bits: int = TEMP_DISC
    """the number of bits per input dimension for the thermometer"""
    n : int = 6
    """ Number of LUT inputs"""
    
    # QUANT
    n_bit_quantization: Optional[int] = 8
    """Quantization of the core network (all expect input/output)"""
    initial_quantization: Optional[int] = 8
    """Quantization of the floating point inputs, bit-width"""
    last_bit_width: int = 8
    """Last requantiation bit-width"""
    enable_quant_step: int = 0
    """When to enable quantization"""
    ptq_calibration: bool = True
    """If used, warmup can be performed (use always)"""
    
    
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = CtrlRoomTemp()
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = CtrlRoomTemp()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, size=256):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), size)
        self.fc2 = nn.Linear(size, size)
        self.fc_mean = nn.Linear(size, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(size, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class ObsNorm(nn.Module):
    def __init__(self, shape, device, eps=1e-8):
        super(ObsNorm, self).__init__()
        mean = torch.zeros(shape, device=device)
        var  = torch.ones(shape, device=device)
        count = torch.tensor(eps, device=device)

        # register the buffers
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)
        self.register_buffer("count", count)

    @torch.no_grad()
    def update(self, x):  # x: [B, obs_dim] tensor
        b = torch.as_tensor(x, device=self.mean.device, dtype=torch.float32)
        batch_mean = b.mean(0)
        batch_var  = b.var(0, unbiased=False)
        batch_count = torch.tensor(b.shape[0], device=self.mean.device, dtype=torch.float32)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, M2 / tot, tot

    def norm(self, x):
        x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        # clamp in -10, 10
        return torch.clamp(x, -10, 10)

def clamp_wnn_params_(actor):
    with torch.no_grad():
        wnn = actor.fc  # your wnn module
        if hasattr(wnn, "net"):
            for idx in (1, 2):
                if idx < len(wnn.net) and hasattr(wnn.net[idx], "luts"):
                    wnn.net[idx].luts.clamp_(-1.0, 1.0)


    
class WNNActor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        from thermometer import ThermometerUniform 
        bits = args.bits
        obs_dim = np.array(env.single_observation_space.shape).prod()
        act_dim = np.prod(env.single_action_space.shape)
        
        if args.network_type == "wnn":
            thermo = ThermometerUniform(n_bits=bits, device='cuda')

            min_values = torch.ones((obs_dim,)) * MIN_TEMP
            max_values = torch.ones((obs_dim,)) * MAX_TEMP
            
            thermo.fit(torch.zeros((1, obs_dim)), min_value=min_values, max_value=max_values)
            self.fc = WNN(obs_dim=np.array(env.single_observation_space.shape).prod(), 
                            act_dim=np.prod(env.single_action_space.shape) * CTRL_DISC, 
                            sizes=[args.size] * args.l, 
                            thermometer=thermo, bits=bits,
                            n=args.n,
                            later_learnable=True,
                            regression=False,
                            )
        else:
            self.fc_mean = None
            
        # network for the log-std-head (dropped at inference time)
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc_logstd = nn.Linear(64, np.prod(env.single_action_space.shape))

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        

    def forward(self, x_in):
        # log_std head
        return self.fc(x_in)
        x = F.tanh(self.fc1(x_in))
        log_std = self.fc_logstd(x)
        
        # Main WNN
        mean = self.fc_mean(x_in)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, deterministic=False):
        # mean, log_std = self(x)
        # std = log_std.exp()            
        # normal = torch.distributions.Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # y_t = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias
        # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        # log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # return action, log_prob, mean
        action_prob = self(x)
        # action_prob: [B, CTRL_DISC], rows sum to 1
        if (action_prob.isnan().any()).any():
            print(f"actoins : {action_prob}, {action_prob.sum()}")
        bucket_idx = action_prob.multinomial(num_samples=1)  # [B,1]
        #bucket_idx = bucket_idx.unsqueeze(1)  # [B, 1]
        action = bucket_idx.float() * CTRL_INTERVAL + 0.5 * CTRL_INTERVAL
        log_prob = torch.log(action_prob.gather(1, bucket_idx) + 1e-8)  # [B,1]

        max_idx = action_prob.argmax(dim=1)
        mean = max_idx.float() * CTRL_INTERVAL + 0.5 * CTRL_INTERVAL
        # if action_prob.dim() == 1:
        #     action_prob = action_prob.unsqueeze(0)

        # bucket_centers = (
        #     0.5 * CTRL_INTERVAL
        #     + CTRL_INTERVAL * torch.arange(
        #         action_prob.shape[1],
        #         device=action_prob.device,
        #         dtype=action_prob.dtype,
        #     )
        # ).unsqueeze(0).expand_as(action_prob)

        # mean = (bucket_centers * action_prob).sum(dim=1)
        return action, log_prob, mean
    

class ActorQuant(WNNActor):
    def __init__(self, env, args, n_bits, initial_quantization):
        super().__init__(env, args)
        self.n_bits = n_bits
        self.initial_quantization = initial_quantization
        assert self.n_bits is not None, "n_bits has to be set for quant."
        self.fc_mean = QuantizedModel(obs_actor=env.single_observation_space.shape[0], 
                                    act=env.single_action_space.shape[0], hidden=[args.size, args.size],
                                    activation_fn=QuantReLU, 
                                    act_bit_width=n_bits, weight_bit_width=n_bits, initial_quantization=initial_quantization, 
                                    last_bit_width=args.last_bit_width,
                                    thermometer=None, 
                                    squash=False,
                                    ptq_calibration=args.ptq_calibration,
                                    )

        # non quantized log-std-head
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)
        self.fc_logstd = nn.Linear(64, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def enable_quant(self):
        self.fc_mean.enable_quant()
    def disable_quant(self):
        self.fc_mean.disable_quant()
   

@torch.no_grad()
def evaluate_actor(
    actor: nn.Module,
    env_id: str,
    normalizer: nn.Module,
    device: torch.device,
    episodes: int = 10,
    seed: int = 123,
    deterministic: bool = True,
    record_stats: bool = True,
    global_step = None,
    writer=None,
):
    eval_env = CtrlRoomTemp()
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)

    actor.eval()  # important: turn off dropout, etc.
    returns = []
    lengths = []

    for ep in range(episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0

        while not (done or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if deterministic:
                
                # use mean action for SAC evaluation
                _, _, mean = actor.get_action(obs_t)
                action = mean[0].cpu().numpy().astype(np.float32)
            else:
                action, _, _ = actor.get_action(obs_t)
                action = action[0].cpu().numpy().astype(np.float32)

            obs, r, done, truncated, infos = eval_env.step(action)
            ep_ret += float(r); ep_len += 1
            if done or truncated:
                returns.append(infos["episode"]["r"][0]); lengths.append(infos["episode"]["l"][0])

    eval_env.close()
    actor.train()  # restore
    
    ret_mean = float(np.mean(returns)); ret_std = float(np.std(returns))
    len_mean = float(np.mean(lengths))

    if record_stats and writer is not None:
        writer.add_scalar("eval/episodic_return_mean", ret_mean, global_step or 0)
        writer.add_scalar("eval/episodic_return_std", ret_std, global_step or 0)
        writer.add_scalar("eval/episodic_length_mean", len_mean, global_step or 0)

    return ret_mean, ret_std, returns, lengths

from tqdm import tqdm
if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"{args.bits}_{args.size}_{args.env_id}_{args.exp_name}_{args.network_type}_{args.initial_quantization}_{args.n_bit_quantization}_{args.last_bit_width}_{'_n=' + str(args.n)}",
        )
    writer = SummaryWriter(f"{args.save_path}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    
    actor = WNNActor(envs, args).to(device)
        
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-5)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs_shape = np.array(envs.single_observation_space.shape).prod()
    
    if not(args.running_normalization):
        obs_norm = None
    else:
        obs_norm = ObsNorm(obs_shape, device=device)


    # evaluate_actor(actor, args.env_id, obs_norm, device=device, writer=writer, global_step=0)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    if obs_norm is not None:
        obs_norm.update(torch.tensor(obs, device=device))

    for global_step in tqdm(range(args.total_timesteps)):
        if global_step == args.enable_quant_step and args.network_type=='quant': #args.total_timesteps // 2:
            if hasattr(actor, "enable_quant"):
                print('enabled quantization')
                actor.enable_quant()
            # check if attribute exists

                
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            if obs_norm is not None:
                obs_t = obs_norm.norm(obs_t)
            actions, _, _ = actor.get_action(obs_t)
            actions = actions.detach().cpu().numpy().astype(np.float32)
        
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
        if obs_norm is not None:
            obs_norm.update(next_obs_t)


        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            if obs_norm is not None:
                o  = obs_norm.norm(data.observations)
                no = obs_norm.norm(data.next_observations)
            else:
                o  = data.observations
                no = data.next_observations
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(no)
                # print(f"next state action {next_state_actions}")
                #print(f"observations {no}")
                # print(f"Sampling random actions for {args.learning_starts} steps: {actions}")
                qf1_next_target = qf1_target(no, next_state_actions)
                qf2_next_target = qf2_target(no, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(o, data.actions).view(-1)
            qf2_a_values = qf2(o, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()
            #print(f"policy frequency: {args.policy_frequency}")
            #counter = -1
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(o)
                    qf1_pi = qf1(o, pi)
                    qf2_pi = qf2(o, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    # check if its an WNN actor and clamp the parameters
                    if isinstance(actor.fc, WNN):
                        clamp_wnn_params_(actor)  # clamp wnn parameters to [-1, 1]

                    if args.autotune:
                        with torch.no_grad():
                            counter += 1
                            #print(f"\n NEW ROUND {counter} \n")
                            a_p = actor(o)
                            #print(f"action prob: {a_p.shape}, negative \n {(a_p < 0).any()} \n NaN {a_p.isnan().any()}")
                            _, log_pi, _ = actor.get_action(o)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 10000 == 0:
                # LOGGING
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                # extract from the wnn the log_alpha parameter and log as scale
                if hasattr(actor.fc_mean, "net") and hasattr(actor.fc_mean.net[-1], "log_alpha"):
                    writer.add_scalar("extras/scale", actor.fc_mean.net[-1].log_alpha.detach().mean().item(), global_step)

                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
        eval_every = args.total_timesteps // 20
        if global_step % eval_every == 0:
           evaluate_actor(actor, args.env_id, obs_norm, device=device, writer=writer, global_step=global_step+1, episodes=10)
        # if last step, do final eval for 1000 episodes
        if global_step == args.total_timesteps - 1:
            evaluate_actor(actor, args.env_id, obs_norm, device=device, writer=writer, global_step=global_step+1, episodes=args.final_evaluation_episodes)
    
    
    envs.close()
    writer.close()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    if args.network_type == 'wnn':

        torch.save({
            "fc_mean": actor.fc_mean.state_dict(),
            "obs_norm": obs_norm.state_dict()
        }, f"{args.save_path}/model_sac_{args.env_id}_{args.size}_{args.bits}_{args.seed}_{args.n}.pth")
    if args.network_type == 'float':
        if obs_norm is None:
            obs_norm = ObsNorm(obs_shape, device=device)
        torch.save({
            "actor": actor.state_dict(),
            "obs_norm": obs_norm.state_dict()
        }, f"{args.save_path}/model_sac_float_{args.env_id}_{args.size}_{args.seed}.pth")
        print('saved float model')
    if args.network_type == 'quant':
        torch.save({
            "actor": actor.state_dict(),
            "obs_norm": obs_norm.state_dict()
        }, f"{args.save_path}/model_sac_quant_{args.env_id}_{args.size}_{args.n_bit_quantization}_{args.initial_quantization}_{args.last_bit_width}_{args.seed}.pth")
        print('saved quant model')