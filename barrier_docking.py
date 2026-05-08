import os
from time import time

import torch
from torch import nn
from torch import Tensor
import numpy as np
import numpy as np
from torch import nn
#from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import tqdm
from conversion import DWN_to_logic_layers
from difflogic_verify.utils.verification_utils import encode_sat_model_with_vars_base_layers
import z3
import sympy as sp
from thermometer import ThermometerUniform
from wnn_models import WNN
import time


"""
Best one so far:
    2 layers of 2500/2000 without the gradient passing to the controller it got to loss of 0.03 where as in other scenarios the avg loss moved around 0.15
    2000 could be better because it reaches 0.05 and has shorter training time
        With gradient passing gets to 0.03 and is stable around it
    Making it wider works better
    Depth had negative effect
    size of lut didnt have much effect

TO DO:
- Compare it with a normal nn
- Increase the lut size
- Gradient passing through to the controller
- Plotting the different dimensions for h and also the velocity constraint as a substraction of the sides of inequality

- Make the allowable position smaller than pos_limit, for example safe_pos = 4 and Init_limit = 5
- Add a buffer between initial and unsafe set, change that in the initial_set z3 funciton as well OR remove the distance constraint from the unsafe set and have it only be the velocity limit.
"""

vel_limit = 0.5
pos_limit = 6 #6
safe_pos = pos_limit - 1 #- 1
starting_pos_limit = safe_pos - 1
N_BITS = 151
VEL_INTERVAL = (2 * vel_limit) / N_BITS
POS_INTERVAL = (2 * pos_limit) / N_BITS
N_SAMPLE_TOTAL = 1000000
#train_model(safe_pos, pos_limit, 0.5, cur_train_file, cur_val_file, cur_model_file, cur_controller_file, threshold, initial_controller_file)
# def train_model(st_pos, unsafe_pos, vel_limit, out_train_file, out_val_file, out_model_file, out_controller_file, threshold, initial_controller_file,  lr=1e-3):
#     model = TwoDimDocking(st_pos, unsafe_pos, vel_limit)
#     V = LyapunovNetworkV(model)
#     datamodule = SampleData(out_train_file, out_val_file, model)
#     controller = Controller(file_name = initial_controller_file, isInitial = True)
#     trainer = Trainer(model, V, controller, datamodule, out_model_file, out_controller_file, threshold, primal_learning_rate=lr)
#     pltrainer = pl.Trainer(max_epochs=10000, accelerator='cpu', callbacks = [EarlyStopping(monitor="saved_loss", patience = 0, mode = 'max', verbose = True)])
#     pltrainer.fit(trainer)

class LyapunovNetworkV(nn.Module):
    def __init__(self, two_dim_docking):
        super().__init__()
        self.two_dim_docking = two_dim_docking
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4,30),
            nn.ReLU(),
            nn.Linear(30,30),
            nn.ReLU(),
            nn.Linear(30,1),
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class TwoDimDocking():
    def __init__(self, st_pos, unsafe_pos, vel_limit):
        n = 0.001027
        self.v0 = 0.2
        self.v1 = 2*n
        #max_dist = math.sqrt(math.pow(st_pos,2) + math.pow(st_pos,2))
        #self.st_vel_limit = round(self.v0 + self.v1*max_dist,4)
        self.st_vel_limit = 0
        self.vel_limit = vel_limit
        self.st_pos = st_pos
        self.unsafe_pos = unsafe_pos

    def unsafe_mask(self,x):
        #unsafe_mask = (abs(x[:,2]) + abs(x[:,3])) > (self.v0 + torch.max(abs(x[:,0]),abs(x[:,1])) * self.v1)
        unsafe_mask = x[:, 2:].norm(dim=-1, p=2) > self.v0+self.v1*(x[:, :2].norm(dim=-1, p=2))
        # unsafe_mask.logical_or_(abs(x[:,0]) >= self.unsafe_pos)
        # unsafe_mask.logical_or_(abs(x[:,1]) >= self.unsafe_pos)
        # unsafe_mask.logical_or_(abs(x[:,2]) >= self.vel_limit)
        # unsafe_mask.logical_or_(abs(x[:,3]) >= self.vel_limit)
        return unsafe_mask
    
    #Same as SimpleDockingInitializer in config_aero.py
    def initial_mask(self,x):
        safe_mask = abs(x[:,2]) == self.st_vel_limit
        safe_mask.logical_and_(abs(x[:,3]) == self.st_vel_limit)

        horizontal_mask = abs(x[:,0]) <= self.unsafe_pos
        horizontal_mask.logical_and_(abs(x[:,0]) >= self.st_pos)
        vertical_mask = abs(x[:,1]) <= self.unsafe_pos
        vertical_mask.logical_and_(abs(x[:,1]) >= self.st_pos)
        position_mask = horizontal_mask.logical_or_(vertical_mask)
        safe_mask.logical_and_(position_mask)
        
        # safe_mask.logical_and_(abs(x[:, 1]) >= 0.35)
        # safe_mask.logical_and_(abs(x[:,0]) >= 0.35)
        #safe_mask.logical_and_(self.nongoal_mask(x))
        return safe_mask
    
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
class Controller():
    def __init__(self, file_name = "fixed_controller_20n_manhattan.pt", t=1, device="cuda"):
        self.file_name = file_name
        self.device = device

        m = 12
        n = 0.001027

        #Q: What is t?
        self.t = t
        
        # self.nn = LearnedController()
        # self.nn.load_state_dict(torch.load(file_name))
        # self.nn = self.nn.to(device="cuda")
        
        # obs_dim = 4
        # thermo = ThermometerUniform(n_bits=N_BITS, device='cuda')

        # min_values = torch.ones((obs_dim//2,)) * pos_limit #* 1.5
        # min_values = torch.cat((-min_values, -torch.ones((obs_dim//2,)) * vel_limit )) #* 1.5
        # max_values = torch.ones((obs_dim//2,)) * pos_limit #* 1.5
        # max_values = torch.cat((max_values, torch.ones((obs_dim//2,)) * vel_limit )) #* 1.5
        
        # thermo.fit(torch.zeros((1, obs_dim)), min_value=min_values, max_value=max_values)
        # self.nn = WNN(obs_dim=obs_dim, 
        #                         act_dim=1, 
        #                         sizes=[2048] * 2, #2500 
        #                         thermometer=thermo, bits=N_BITS,
        #                         n=2,
        #                         later_learnable=True).to(device)
        

        # checking a more elaborate inductive property holds (closer or velocity decreases in appropriate direction)

        # velocity expansion is arbitrary

        # Matrix encoding of system dynamics
        self.coeffs_x_t = [
            4 - 3 * np.cos(n * t),
            0,
            1 / n * np.sin(n * t),
            2 / n - 2 / n * np.cos(n * t),
            (1 - np.cos(n * t)) / (m * n ** 2),
            2 * t / (m * n) - 2 * np.sin(n * t) / (m * n ** 2),
        ]
        self.coeffs_y_t = [
            -6 * n * t + 6 * np.sin(n * t),
            1,
            -2 / n + 2 / n * np.cos(n * t),
            -3 * t + 4 / n * np.sin(n * t),
            (-2 * t) / (m * n) + (2 * np.sin(n * t)) / (m * n ** 2),
            4 / (m * n ** 2) - (3 * t ** 2) / (2 * m) - (4 * np.cos(n * t)) / (m * n ** 2),
        ]
        self.coeffs_v_x_t = [
            3 * n * np.sin(n * t),
            0,
            np.cos(n * t),
            2 * np.sin(n * t),
            np.sin(n * t) / (m * n),
            2 / (m * n) - (2 * np.cos(n * t)) / (m * n),
        ]
        self.coeffs_v_y_t = [
            -6 * n + 6 * n * np.cos(n * t),
            0,
            -2 * np.sin(n * t),
            -3 + 4 * np.cos(n * t),
            (2 * np.cos(n * t) - 2) / (m * n),
            (-3 * t) / (m) + (4 * np.sin(n * t)) / (m * n),
        ]
        self.coeff_arr = torch.Tensor([self.coeffs_x_t,self.coeffs_y_t,self.coeffs_v_x_t,self.coeffs_v_y_t]).to(device=self.device)
        #self.coeff_arr = self.coeff_arr.to(device="cuda")
    
    def next_step(self, x):
        forces_not_clipped = self.nn.forward(x)
        #forces = torch.tanh(forces_not_clipped) # forces are between -1 and 1, but the output of the nn is unbounded, so we use atanh to map it to the real line
        forces = torch.clip(forces_not_clipped,-1,1)
        
        total_input = torch.cat((x,forces),1)

        next_step = torch.mm(total_input, torch.transpose(self.coeff_arr,0,1))

        return next_step
    

class SampleData():
    def __init__(self, two_dim_docking, num_points = N_SAMPLE_TOTAL, dim = 4, val_split = 0.1, batch_size = 1000, max_tries = 5000):
        self.num_points = num_points
        self.two_dim_docking = two_dim_docking
        self.dim = dim
        self.val_split = val_split
        self.batch_size = batch_size
        self.max_tries = max_tries
        self.ranges = []

        for _ in range(2):
            self.ranges.append([-self.two_dim_docking.unsafe_pos-0.2, self.two_dim_docking.unsafe_pos+0.2])

        for _ in range(2):
            #ranges.append([-self.vel_limit,self.vel_limit])
            self.ranges.append([-self.two_dim_docking.vel_limit-0.05,self. two_dim_docking.vel_limit+0.05])
        
    def prepare_data(self):
        x = torch.Tensor(self.num_points*4//5, self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(self.dim):
            min_val, max_val = self.ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

        y = torch.Tensor(self.num_points//5, self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(2):
            y[:, i] = y[:, i] * (self.two_dim_docking.st_pos - (-self.two_dim_docking.st_pos)) + (-self.two_dim_docking.st_pos)

        for j in range(2):
            y[:, j+2] = y[:, j+2] * (self.two_dim_docking.st_vel_limit - (-self.two_dim_docking.st_vel_limit)) + (-self.two_dim_docking.st_vel_limit)

        x = torch.cat((x,y))
        
        random_indices = torch.randperm(len(x))

        self.x_train = x[random_indices]

        
        x = torch.Tensor(self.num_points*4//50, self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(self.dim):
            min_val, max_val = self.ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

        y = torch.Tensor(self.num_points//50, self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(2):
            y[:, i] = y[:, i] * (self.two_dim_docking.st_pos - (-self.two_dim_docking.st_pos)) + (-self.two_dim_docking.st_pos)

        for j in range(2):
            y[:, j+2] = y[:, j+2] * (self.two_dim_docking.st_vel_limit - (-self.two_dim_docking.st_vel_limit)) + (-self.two_dim_docking.st_vel_limit)

        x = torch.cat((x,y))

        random_indices = torch.randperm(len(x))
        self.x_val = x[random_indices]

    
def train_controller_and_certificate(controller, certificate_nn, data, load_model = False, model_path = "ppo_docking_1000000_steps.pt"):
    """
    Function for training the controller and certificate. The function samples points from the safe set, unsafe set and the rest of the state space and uses them to compute the loss function for training the controller and certificate. The function returns the trained controller and certificate.
    """
  
    device = "cuda" 
    print("starting fine-tuning of the controller and certificate")
    # obs_dim = 4
    # thermo = ThermometerUniform(n_bits=N_BITS, device='cuda')

    # min_values = torch.ones((obs_dim//2,)) * pos_limit #* 1.5
    # min_values = torch.cat((-min_values, -torch.ones((obs_dim//2,)) * vel_limit )) #* 1.5
    # max_values = torch.ones((obs_dim//2,)) * pos_limit #* 1.5
    # max_values = torch.cat((max_values, torch.ones((obs_dim//2,)) * vel_limit )) #* 1.5
    
    # thermo.fit(torch.zeros((1, obs_dim)), min_value=min_values, max_value=max_values)
    # certificate_nn = WNN(obs_dim=obs_dim, 
    #                         act_dim=1, 
    #                         sizes=[2000] * 2, #2500 
    #                         thermometer=thermo, bits=N_BITS,
    #                         n=2,
    #                         later_learnable=True).to(device)
    #certificate_nn = LyapunovNetworkV(TwoDimDocking(starting_pos_limit, safe_pos, vel_limit)).to(device)
    
    controller_nn = Controller()
    controller_nn.nn = controller
    if load_model:
        from types import SimpleNamespace
        from ppo_train_docking import make_env
        from ppo_train_docking import WNNActor
        eval_env = make_env(
            "Docking2d",
            idx=0,
            capture_video=False,
            run_name="controller_certificate_fine_tuning",
            gamma=0.99,
        )()
        env_view = SimpleNamespace(
            single_observation_space=eval_env.observation_space,
            single_action_space=eval_env.action_space,
        )
        controller = WNNActor.from_checkpoint(env=env_view, path=str(model_path), device=device)
        controller.use_tanh_final = True
        controller.to(device)
    #controller_nn.nn.to(device)
    # data = SampleData(TwoDimDocking(starting_pos_limit, safe_pos, vel_limit))
    # print("preparing data...")
    # data.prepare_data()
    # print("data prepared, starting training...")
    points = data.x_train.to(device)
    certificate_nn.train()

    # Hyperparameters
    batch_size = data.batch_size
    lr = 1e-2
    gamma = 1

    x_all = points #torch.tensor(points, device=device, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(list(certificate_nn.parameters()), lr=lr)
    optimizer_both = torch.optim.Adam(list(certificate_nn.parameters()) + list(controller_nn.nn.parameters()), lr=lr/10)


    history = []

    # Train until the loss is (numerically) zero
    max_epochs = 10000
    tol = 1e-8
    epoch = 0

    for epoch in tqdm.tqdm(range(max_epochs)):
        perm = torch.randperm(x_all.shape[0], device=device)
        epoch_loss = 0.0

        for start in range(0, x_all.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = x_all[idx]  # (B, 1)
            loss = 0
            zero = torch.tensor(0.0, device=device)
            initial_mask =  data.two_dim_docking.initial_mask(xb)
            unsafe_mask = data.two_dim_docking.unsafe_mask(xb)
            #safe_mask = ~unsafe_mask 
            # L = (1/N_I) * SUM(m(-h(d))| d∈D_I) + (1/N_U) * SUM(m(h(d))| d∈D_U) + (1/N) * SUM(m(-∆h (xk, uk) - γ.h(xk))| d∈Internal(S))
            #u = controller_nn.nn(xb)  # (B, 1)
            h_x = certificate_nn(xb)  # (B, 1)
            interior_h = (h_x > 0) 
            x_next = controller_nn.next_step(xb)  # (B, 1)
            h_next = certificate_nn(x_next)  # (B, 1)
            delta_h = h_next - h_x
            loss_i = F.relu(-h_x[initial_mask]).sum() / initial_mask.sum() if initial_mask.any() else zero
            loss_u = F.relu(h_x[unsafe_mask]).sum() / unsafe_mask.sum() if unsafe_mask.any() else zero
            loss_d = F.relu(-delta_h - gamma * h_x)[interior_h].sum() / interior_h.sum() if interior_h.any() else zero
            loss += loss_i + loss_u + loss_d
            if epoch > 20:
                optimizer_both.zero_grad()
                loss.backward()
                optimizer_both.step()
                # if start >= x_all.shape[0] - batch_size - 1:
                #     for name, p in controller_nn.nn.named_parameters():
                #         if p.grad is not None:
                #             print("controller.", name, p.grad.shape ,p.grad.abs().mean().item())
                #     for name, p in certificate_nn.named_parameters():
                #         if p.grad is not None:
                #             print("certificate.", name, p.grad.shape ,p.grad.abs().mean().item())
                #     # for gi, group in enumerate(optimizer_both.param_groups):
                #     #     for p in group["params"]:
                #     #         if p.grad is not None:
                #     #             print(gi, p.shape, p.grad.norm().item())
            else:
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(
                #     list(controller_nn.parameters()) + list(certificate_nn.parameters()), 1.0
                # )
                optimizer.step()

            epoch_loss += float(loss.detach().cpu())

        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

        history.append(epoch_loss)

        # Stop when loss is effectively zero (or if max epochs reached)
        if epoch_loss <= tol: #or epoch_loss <= tol
            print(f"Stopping training at epoch {epoch} with loss {epoch_loss:.4f}")
            break
    print(f"Final Epoch {epoch}, Loss: {epoch_loss:.4f}")
    print("finished fine-tuning of the controller and certificate")
    return controller_nn.nn, certificate_nn


#TO DO: Make sure all the constraints through out the computation are returned
def transiton_function_bitvector(state_bits, control_bits, postfix="prime"):
    counts = [z3.Sum([z3.If(b, 1, 0) for b in s]) for s in state_bits + control_bits] 
    i_x, i_y, i_vx, i_vy, i_ux, i_uy = counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
    # ==========================
    # Fixed configuration
    # ==========================
    _N = 0.001027
    _M = 12
    _T = 1.0
    _SCALE = 1 << 20
    _BW = 64

    # Input domains for [x, y, vx, vy, ux, uy]
    _DOMAINS_IN = [
        (-6.0, 6.0, 100),
        (-6.0, 6.0, 100),
        (-0.5, 0.5, 100),
        (-0.5, 0.5, 100),
        (-1.0, 1.0, 100),
        (-1.0, 1.0, 100),
    ]

    # Output domains for [x', y', vx', vy']
    _DOMAINS_OUT = [
        (-6.0, 6.0, 100),
        (-6.0, 6.0, 100),
        (-0.5, 0.5, 100),
        (-0.5, 0.5, 100),
    ]

    # Coefficients in output order: [fx, fy, fvx, fvy]
    _COEFF_ROWS = [
        [
            4 - 3 * np.cos(_N * _T),
            0.0,
            np.sin(_N * _T) / _N,
            2 / _N - 2 * np.cos(_N * _T) / _N,
            (1 - np.cos(_N * _T)) / (_M * _N**2),
            2 * _T / (_M * _N) - 2 * np.sin(_N * _T) / (_M * _N**2),
        ],
        [
            -6 * _N * _T + 6 * np.sin(_N * _T),
            1.0,
            -2 / _N + 2 * np.cos(_N * _T) / _N,
            -3 * _T + 4 * np.sin(_N * _T) / _N,
            -2 * _T / (_M * _N) + 2 * np.sin(_N * _T) / (_M * _N**2),
            4 / (_M * _N**2) - (3 * _T**2) / (2 * _M) - 4 * np.cos(_N * _T) / (_M * _N**2),
        ],
        [
            3 * _N * np.sin(_N * _T),
            0.0,
            np.cos(_N * _T),
            2 * np.sin(_N * _T),
            np.sin(_N * _T) / (_M * _N),
            2 / (_M * _N) - 2 * np.cos(_N * _T) / (_M * _N),
        ],
        [
            -6 * _N + 6 * _N * np.cos(_N * _T),
            0.0,
            -2 * np.sin(_N * _T),
            -3 + 4 * np.cos(_N * _T),
            (2 * np.cos(_N * _T) - 2) / (_M * _N),
            -3 * _T / _M + 4 * np.sin(_N * _T) / (_M * _N),
        ],
    ]

    def _sbv_const(v: int, bw: int = _BW):
        return z3.BitVecVal(v % (1 << bw), bw)

    def _precompute_A_B_row(coeff_row):
        lo = [d[0] for d in _DOMAINS_IN]
        step = [(d[1] - d[0]) / d[2] for d in _DOMAINS_IN]
        A_row = [int(round(coeff_row[d] * step[d] * _SCALE)) for d in range(6)]
        B_row = int(round(sum(coeff_row[d] * lo[d] for d in range(6)) * _SCALE))
        return A_row, B_row

    # A is 4x6, B is length-4 vector
    A_rows = []
    B_vec = []
    for coeff_row in _COEFF_ROWS:
        A_row, B_row = _precompute_A_B_row(coeff_row)
        A_rows.append(A_row)
        B_vec.append(B_row)

    def transition_function_bitvector_int_io(i_x, i_y, i_vx, i_vy, i_ux, i_uy):
        # Inputs are Int indices, expected constrained to [0,99].
        # Use named BV variables (not anonymous Int2BV casts) so they are
        # inspectable in a counterexample model and reusable across constraints.
        input_names = ['x', 'y', 'vx', 'vy', 'ux', 'uy']
        idx_bv = [z3.BitVec(f'idx_bv_{name}', _BW) for name in input_names]
        int_vals = [i_x, i_y, i_vx, i_vy, i_ux, i_uy]
        n_bins_in = [d[2] for d in _DOMAINS_IN]

        all_constraints = []

        # Bind each named BV variable to the Int count via equality, and assert
        # the unsigned upper bound so the BV solver can prune the search space.
        for j, (bv_var, int_val, nb) in enumerate(zip(idx_bv, int_vals, n_bins_in)):
            all_constraints.append(bv_var == z3.Int2BV(int_val, _BW))
            all_constraints.append(z3.ULE(bv_var, _sbv_const(nb - 1)))

        out_idx_int = []

        for k, (lo, hi, n_bins) in enumerate(_DOMAINS_OUT):
            lo_s = int(round(lo * _SCALE))
            hi_s = int(round(hi * _SCALE))
            range_s = int(round((hi - lo) * _SCALE))

            lo_bv = _sbv_const(lo_s)
            hi_bv = _sbv_const(hi_s)
            n_bv = _sbv_const(n_bins)
            range_bv = _sbv_const(range_s)
            zero_bv = _sbv_const(0)
            max_bv = _sbv_const(n_bins - 1)

            # Named variable for the scaled linear combination f_k so it is
            # inspectable in the solver model and only computed once.
            f_bv = z3.BitVec(f'f_bv_{k}', _BW)
            f_bv_expr = _sbv_const(B_vec[k])
            for j in range(6):
                a_kj = A_rows[k][j]
                if a_kj != 0:
                    f_bv_expr = f_bv_expr + _sbv_const(a_kj) * idx_bv[j]
            all_constraints.append(f_bv == f_bv_expr)

            # Named variables for each intermediate BV result so the solver
            # can inspect them and they can be referenced in multiple places.
            shifted_bv = z3.BitVec(f'shifted_bv_{k}', _BW)
            all_constraints.append(shifted_bv == f_bv - lo_bv)

            numer_bv = z3.BitVec(f'numer_bv_{k}', _BW)
            all_constraints.append(numer_bv == shifted_bv * n_bv)

            idx_raw_bv = z3.BitVec(f'idx_raw_bv_{k}', _BW)
            # Use signed division (BitVec '/' is bvsdiv in z3py)
            all_constraints.append(idx_raw_bv == (numer_bv / range_bv))

            idx_out_bv = z3.BitVec(f'idx_out_bv_{k}', _BW)
            # Use signed comparisons since f_bv, lo_bv, hi_bv are two's complement fixed-point values
            # BitVec '<=', '>=', '>' are signed in z3py. Use U* variants for unsigned.
            below = f_bv <= lo_bv
            above = f_bv >= hi_bv
            idx_mid = z3.If(idx_raw_bv > max_bv, max_bv, idx_raw_bv)
            all_constraints.append(
                idx_out_bv == z3.If(below, zero_bv, z3.If(above, max_bv, idx_mid))
            )

            idx_int = z3.BV2Int(idx_out_bv, is_signed=False)
            out_idx_int.append(idx_int)

        return tuple(out_idx_int), all_constraints

    def index_to_thermometer(idx, name):
        output_vars = [z3.Bool(f'{name}_{postfix}_{i}') for i in range(N_BITS)]
        def rec_cond(k): 
            if k == 0: 
                return z3.If(idx == 0, z3.And([output_vars[i] == True for i in range(k+1)] + [output_vars[i] == False for i in range(k+1, N_BITS)]), False)
            return z3.If(idx == k, z3.And([output_vars[i] == True for i in range(k+1)] + [output_vars[i] == False for i in range(k+1, N_BITS)]), rec_cond(k-1))
        
        return rec_cond(N_BITS -1), output_vars


    (out_x, out_y, out_vx, out_vy), transition_constraints = transition_function_bitvector_int_io(i_x, i_y, i_vx, i_vy, i_ux, i_uy)
    out_x_cond, out_x_vars = index_to_thermometer(out_x, "x")
    out_y_cond, out_y_vars = index_to_thermometer(out_y, "y")
    out_vx_cond, out_vx_vars = index_to_thermometer(out_vx, "vx")
    out_vy_cond, out_vy_vars = index_to_thermometer(out_vy, "vy")
    return z3.And(out_x_cond, out_y_cond, out_vx_cond, out_vy_cond), [out_x_vars, out_y_vars, out_vx_vars, out_vy_vars], transition_constraints


def create_norm_map_vel_limit(i_x, i_y):
    """
    Maps discretized velocity indices to the norm (x^2 + y^2)^0.5.
    
    Input domain: x, y each in [-vel_limit, vel_limit] with 100 intervals
    Output domain: [0.2, 0.2 + 2 * 0.001027 * (pos_limit * sqrt(2)) * 1.5] with 100 intervals
    
    Args:
        i_x: Index in [0, 99] for x dimension
        i_y: Index in [0, 99] for y dimension
    
    Returns:
        Tuple of (output_index, lower_bound, upper_bound) where output_index is in [0, 99]
    """
    _VEL_LIMIT = vel_limit
    _N = 0.001027
    _V0 = 0.2
    _V1_COEFF = 2 * _N
    _N_BINS = N_BITS
    
    # Map indices to actual values (bin boundaries)
    x_step = (2 * _VEL_LIMIT) / _N_BINS
    y_step = (2 * _VEL_LIMIT) / _N_BINS
    
    # Get lowest and highest values x, y can take in their intervals
    x_min = -_VEL_LIMIT + i_x * x_step
    x_max = -_VEL_LIMIT + (i_x + 1) * x_step
    y_min = -_VEL_LIMIT + i_y * y_step
    y_max = -_VEL_LIMIT + (i_y + 1) * y_step
    
    # Compute norm at all corners of the rectangle to find min and max
    norms = [
        np.sqrt(x_min**2 + y_min**2),
        np.sqrt(x_min**2 + y_max**2),
        np.sqrt(x_max**2 + y_min**2),
        np.sqrt(x_max**2 + y_max**2)
    ]
    norm_min = min(norms)
    norm_max = max(norms)
    
    # Output range: [V0, V0 + V1_COEFF * (pos_limit * sqrt(2)) * 1.5]
    max_norm = (_V0 + _V1_COEFF * (pos_limit * np.sqrt(2))) * 1.5
    norm_step = (max_norm) / _N_BINS
    
    # Compute output indices
    idx_min = int(np.clip(norm_min/ norm_step, 0, _N_BINS - 1))
    idx_max = int(np.clip(norm_max/ norm_step, 0, _N_BINS - 1))
    
    return idx_min, idx_max


def create_velocity_constraint_map_pos_limit(i_x, i_y):
    """
    Maps discretized position indices to the velocity constraint 0.2 + 2 * 0.001027 * (x^2 + y^2)^0.5.
    
    Input domain: x, y each in [-pos_limit, pos_limit] with 100 intervals
    Output domain: [0.2, 0.2 + 2 * 0.001027 * (pos_limit * sqrt(2)) * 1.5] with 100 intervals
    
    Args:
        i_x: Index in [0, 99] for x dimension
        i_y: Index in [0, 99] for y dimension
    
    Returns:
        Tuple of (output_index, lower_bound, upper_bound) where output_index is in [0, 99]
    """
    _POS_LIMIT = pos_limit
    _N = 0.001027
    _V0 = 0.2
    _V1_COEFF = 2 * _N
    _N_BINS = N_BITS
    
    # Map indices to actual values (bin boundaries)
    x_step = (2 * _POS_LIMIT) / _N_BINS
    y_step = (2 * _POS_LIMIT) / _N_BINS
    
    # Get lowest and highest values x, y can take in their intervals
    x_min = -_POS_LIMIT + i_x * x_step
    x_max = -_POS_LIMIT + (i_x + 1) * x_step
    y_min = -_POS_LIMIT + i_y * y_step
    y_max = -_POS_LIMIT + (i_y + 1) * y_step
    
    # Compute norm at all corners of the rectangle to find min and max
    norms = [
        np.sqrt(x_min**2 + y_min**2),
        np.sqrt(x_min**2 + y_max**2),
        np.sqrt(x_max**2 + y_min**2),
        np.sqrt(x_max**2 + y_max**2)
    ]
    norm_min = min(norms)
    norm_max = max(norms)
    
    # Compute velocity constraint values at min and max norms
    constraint_min = _V0 + _V1_COEFF * norm_min
    constraint_max = _V0 + _V1_COEFF * norm_max
    
    # Output range: [V0, V0 + V1_COEFF * (pos_limit * sqrt(2)) * 1.5]
    max_constraint = _V0 + _V1_COEFF * (_POS_LIMIT * np.sqrt(2)) * 1.5
    constraint_step = (max_constraint) / _N_BINS
    
    # Compute output indices
    idx_min = int(np.clip(norm_min / constraint_step, 0, _N_BINS - 1))
    idx_max = int(np.clip(norm_max / constraint_step, 0, _N_BINS - 1))
    
    return idx_min, idx_max

BARRIER_DISC = N_BITS
#Initializeing bounds for barrier: These are just initial values, the actual bounds can be inferred from the alpha and beta parameters in the regression head of the certificate neural network after training; the number of intervals for each of the spaces can be a hyperprameter;
BARRIER_MAX = 3
BARRIER_MIN = -1
BARRIER_INTERVAL = (BARRIER_MAX - BARRIER_MIN) / BARRIER_DISC
CTRL_DISC = N_BITS
CTRL_INTERVAL = (2) / CTRL_DISC
def get_groupsum_layer_output_mapping(model, states, type_wnnactor: bool = False):
    # features = {}
    # def get_features(name):
    #     def hook(model, input, output):
    #         features[name] = output.detach()

    #     return hook
    # # Register a forward hook on the layer before the groupsum layer
    # if not type_wnnactor:
    #     model.net[-2].register_forward_hook(get_features("group_sum"))
    # else:
    #     model.actor_mean.net[-2].register_forward_hook(get_features("group_sum"))
    # device = next(model.parameters()).device
    # states = torch.tensor(states, dtype=torch.float32)
    # states = states.to(device)
    #output = model(states)
    #grps_out = features["group_sum"]
    group_sums = []
    if not type_wnnactor:
        total_nb = int(model.net[-1].norm_factor)   
        group_sums = [[i] for i in range(total_nb+1)]
        output = model.net[-1].forward(torch.tensor(group_sums, dtype=torch.float32).to(device="cuda"))
    else:
        total_nb = int(model.actor_mean.net[-1].norm_factor)
        group_sums = [[i,i] for i in range(total_nb+1)]
        output = model.actor_mean.net[-1].forward(torch.tensor(group_sums, dtype=torch.float32).to(device="cuda"))
        output = torch.tanh(output) # since the output of the controller is between -1 and 1, we use tanh to map it to the correct range; for the certificate we will infer the bounds from the alpha and beta parameters in the regression head after training, so we don't need to apply tanh to it here
        
    
    #The bounds for controller it is beween 0 and 1; for certificate
    #it has to be infered from the alpha and beta parameter in the regression head; the number of intervals for each of the spaces can be a hyperprameter; 
    # BARRIER_DISC for certificate and CTRL_DISC for controller; 
    group_sum_func_map = {}
    if type_wnnactor:
        group_sum_func_map['ux'] = {}
        group_sum_func_map['uy'] = {}
    if not type_wnnactor:
        alpha = torch.exp(model.net[-1].log_alpha.data[0]).cpu().item()
        beta = model.net[-1].beta.data[0].cpu().item()
        BARRIER_MIN = -alpha/2 + beta
        BARRIER_MAX = alpha/2 + beta
        BARRIER_INTERVAL = (BARRIER_MAX - BARRIER_MIN) / BARRIER_DISC
        #Corrction to make sure that zero is the beginning of an interval in the barrier case; this is important for the verification step
        prezero = (0-BARRIER_MIN) // BARRIER_INTERVAL
        BARRIER_INTERVAL += -10e-8 + abs(prezero * BARRIER_INTERVAL + BARRIER_MIN) / prezero #subtracted epsilon for numerical stability
        BARRIER_MAX = BARRIER_MIN + BARRIER_INTERVAL * BARRIER_DISC
    for i, gs in enumerate(group_sums):
        if not type_wnnactor:
            group_sum_func_map[i] = int((output[i].cpu() - BARRIER_MIN) // BARRIER_INTERVAL)
        else:
            group_sum_func_map['ux'][i] = int((output[i][0].cpu() + 1) // CTRL_INTERVAL)
            group_sum_func_map['uy'][i] = int((output[i][1].cpu() + 1) // CTRL_INTERVAL)
    
    return group_sum_func_map


def group_sum_func_map_constraint(group_sum_func_map, name, gs_count, output_range=BARRIER_DISC):
    output = [z3.Bool(f'{name}_{i}') for i in range(output_range)]
    def create_output_bool_vec(k):
        return [True if j< group_sum_func_map[k] else False for j in range(output_range)]

    def rec_cond(k): 
        if k == 0:
            if 0 not in group_sum_func_map:
                return z3.BoolVal(False)  
            return z3.If(gs_count == 0, z3.And([output[i] == b for i,b in enumerate(create_output_bool_vec(0))]), False)
        if k not in group_sum_func_map:
            return rec_cond(k-1)
        return z3.If(gs_count == k, z3.And([output[i] == b for i,b in enumerate(create_output_bool_vec(k))]), rec_cond(k-1))
    return rec_cond(output_range), output


def negative_barrier_constraint(h_real_output, barrier_zero_bool_vec):
    h_neg = z3.Or([z3.And(z3.Not(h_real_output[i]), barrier_zero_bool_vec[i]) for i in range(BARRIER_DISC)])
    return h_neg

def check_h_nonneg(h_real_output, barrier_zero_bool_vec):
    h_nonneg = z3.And([z3.Implies(barrier_zero_bool_vec[i], h_real_output[i]) for i in range(BARRIER_DISC)])
    return h_nonneg

def initial_set_constraint(state_bits):
    zero_vel_idx = int((0 - vel_limit) / VEL_INTERVAL)
    start_pos_idx = int((starting_pos_limit + pos_limit) / POS_INTERVAL)
    end_pos_idx = int((safe_pos + pos_limit) / POS_INTERVAL)
    start_neg_idx = int(( -starting_pos_limit + pos_limit) / POS_INTERVAL)
    end_neg_idx = int((-safe_pos + pos_limit) / POS_INTERVAL)
    
    x_bits = state_bits[0]
    y_bits = state_bits[1]
    vx_bits = state_bits[2]
    vy_bits = state_bits[3]
    constraints = [z3.And([x_bits[i] for i in range(start_pos_idx+1)])] # x_bits should be greater than or equal to the start index 
    constraints += [z3.And([z3.Not(x_bits[i]) for i in range(end_pos_idx+1, N_BITS)])]  # x_bits should be less than or equal to the end index
    constraints += [z3.And([x_bits[i] for i in range(end_neg_idx+1)])] # For negative the start and end are reversed
    constraints += [z3.And([z3.Not(x_bits[i]) for i in range(start_neg_idx+1, N_BITS)])]  

    constraints += [z3.And([y_bits[i] for i in range(start_pos_idx+1)])] 
    constraints += [z3.And([z3.Not(y_bits[i]) for i in range(end_pos_idx+1, N_BITS)])]  
    constraints += [z3.And([y_bits[i] for i in range(end_neg_idx+1)])] 
    constraints += [z3.And([z3.Not(y_bits[i]) for i in range(start_neg_idx+1, N_BITS)])]  

    constraints += [z3.And([vx_bits[i] for i in range(zero_vel_idx+1)])]  # vx_bits should be equal to the zero velocity index
    constraints += [z3.And([z3.Not(vx_bits[i]) for i in range(zero_vel_idx+1, N_BITS)])]  
    constraints += [z3.And([vy_bits[i] for i in range(zero_vel_idx+1)])]  
    constraints += [z3.And([z3.Not(vy_bits[i]) for i in range(zero_vel_idx+1, N_BITS)])]  

    return z3.And(constraints)

def unsafe_set_constraint(state_bits):
    x_bits = state_bits[0]
    y_bits = state_bits[1]  
    vx_bits = state_bits[2]
    vy_bits = state_bits[3]

    left_map = {}
    right_map = {}
    for i in range(N_BITS):
        for j in range(N_BITS):
          left_map[(i,j)] = create_norm_map_vel_limit(i,j) 
          right_map[(i,j)] = create_velocity_constraint_map_pos_limit(i,j)
    #Overapproximation of left (velocity) larger than underapproximation of right side (position) of the unsafe constraint
    x_count = z3.Sum([z3.If(b, 1, 0) for b in x_bits])
    y_count = z3.Sum([z3.If(b, 1, 0) for b in y_bits])
    vx_count = z3.Sum([z3.If(b, 1, 0) for b in vx_bits])
    vy_count = z3.Sum([z3.If(b, 1, 0) for b in vy_bits])
    left = z3.Int("left")
    right = z3.Int("right")

    left_constraints = []
    for x_idx in range(1, N_BITS + 1):
        for y_idx in range(1, N_BITS + 1):
            left_constraints.append(z3.Implies(z3.And(x_count == x_idx, y_count == y_idx), left == left_map[(x_idx-1,y_idx-1)][1]))
    right_constraints = []
    for vx_idx in range(1, N_BITS + 1):
        for vy_idx in range(1, N_BITS + 1):
            right_constraints.append(z3.Implies(z3.And(vx_count == vx_idx, vy_count == vy_idx), right == right_map[(vx_idx-1,vy_idx-1)][0]))

    end_constraint = left > right

    return z3.And(left_constraints + right_constraints + [end_constraint])

def sample_points_around_counterexample(counterexample):
    """
    Function for sampling points around a counterexample. This can be used to find more counterexamples in the neighborhood of the initial counterexample returned by the solver, which can help in better understanding the failure cases and improving the training of the controller and certificate.
    The function takes in a counterexample point, a radius for sampling around that point, and the number of points to sample. It returns a list of sampled points around the counterexample.
    """
    radius_pos = (2 * pos_limit) / 15  
    radius_vel = (2 * vel_limit) / 15
    num_points = N_SAMPLE_TOTAL // 10
    x = np.random.normal(counterexample[0], radius_pos , size=(num_points, 1)).astype(np.float32)
    y = np.random.normal(counterexample[1], radius_pos , size=(num_points, 1)).astype(np.float32)
    vx = np.random.normal(counterexample[2], radius_vel , size=(num_points, 1)).astype(np.float32)
    vy = np.random.normal(counterexample[3], radius_vel , size=(num_points, 1)).astype(np.float32)
    #TO DO: Check if the hstack is correct
    all_points = np.hstack((x, y, vx, vy))
    return all_points

"""
Composing the verification formula:
∃x ∈ X :(x ∈ XI ∧ h(x) < 0) ∨ (x ∈ XU ∧ h(x) ≥ 0) ∨ ((x′ = f (x, π(x)) ∧  h(x) >= 0) ∧ (h(x′) - h(x) < − γ.h(x)) )
"""

def verification_loop(controller_nn, load_model = False):
    if load_model:
        from types import SimpleNamespace
        from ppo_train_docking import make_env
        from ppo_train_docking import WNNActor
        eval_env = make_env(
            "Docking2d",
            idx=0,
            capture_video=False,
            run_name="controller_certificate_fine_tuning",
            gamma=0.99,
        )()
        env_view = SimpleNamespace(
            single_observation_space=eval_env.observation_space,
            single_action_space=eval_env.action_space,
        )
        model_path = "BeforeCEGIS_ppo_train_docking.cleanrl_model"
        controller_nn = WNNActor.from_checkpoint(env=env_view, path=str(model_path), device="cuda")
        controller_nn.use_tanh_final = True
        controller_nn.to("cuda")
    device = next(controller_nn.parameters()).device
    data = SampleData(TwoDimDocking(starting_pos_limit, safe_pos, vel_limit))
    print("preparing data...")
    data.prepare_data()
    print("data prepared, starting training...")
    #points = data.x_train.to(device)
    #points = sample_points((MIN_TEMP, MAX_TEMP))
    #lower_bound, upper_bound = lower_upper_transition_function()
    obs_dim = 4
    thermo = ThermometerUniform(n_bits=N_BITS, device='cuda')

    min_values = torch.ones((obs_dim//2,)) * pos_limit #* 1.5
    min_values = torch.cat((-min_values, -torch.ones((obs_dim//2,)) * vel_limit )) #* 1.5
    max_values = torch.ones((obs_dim//2,)) * pos_limit #* 1.5
    max_values = torch.cat((max_values, torch.ones((obs_dim//2,)) * vel_limit )) #* 1.5
    
    thermo.fit(torch.zeros((1, obs_dim)), min_value=min_values, max_value=max_values)
    certificate_nn = WNN(obs_dim=obs_dim, 
                            act_dim=1, 
                            sizes=[2000] * 2, #2500 
                            thermometer=thermo, bits=N_BITS,
                            n=2,
                            later_learnable=True).to(device)

    # print("Sampling input states for the verification formula...")
    # input_states = []
    # pos_intrv = pos_limit/N_BITS
    # vel_intrv = vel_limit/N_BITS
    # # for x_idx in range(N_BITS):
    # #     for y_idx in range(N_BITS):
    # #         for vx_idx in range(N_BITS):
    # #             for vy_idx in range(N_BITS):
    # #                 x = -pos_limit + x_idx * pos_intrv + pos_intrv
    # #                 y = -pos_limit + y_idx * pos_intrv + pos_intrv
    # #                 vx = -vel_limit + vx_idx * vel_intrv + vel_intrv
    # #                 vy = -vel_limit + vy_idx * vel_intrv + vel_intrv
    # #                 input_states.append([x, y, vx, vy])
    # # print(f"Total input states sampled: {len(input_states)}")
    # # input_states = torch.tensor(input_states, dtype=torch.float32)

    # # Generate index arrays
    # x_indices = np.arange(N_BITS)
    # y_indices = np.arange(N_BITS)
    # vx_indices = np.arange(N_BITS)
    # vy_indices = np.arange(N_BITS)
    # # Create a meshgrid for all combinations of indices
    # X_idx, Y_idx, VX_idx, VY_idx = np.meshgrid(x_indices, y_indices, vx_indices, vy_indices, indexing='ij')

    # # Calculate the corresponding values
    # X = -pos_limit + (X_idx + 1) * pos_intrv
    # Y = -pos_limit + (Y_idx + 1) * pos_intrv
    # VX = -vel_limit + (VX_idx + 1) * vel_intrv
    # VY = -vel_limit + (VY_idx + 1) * vel_intrv

    # # Stack them to form the input_states
    # input_states_np = np.stack([X.ravel(), Y.ravel(), VX.ravel(), VY.ravel()], axis=1)

    # print(f"Total input states sampled: {input_states_np.shape[0]}")
    # input_states = torch.tensor(input_states_np, dtype=torch.float32)
    # input_states = input_states.to(device)
    iteration = -1
    while True:
        iteration += 1
        print(f"Starting verification iteration {iteration}...")
        controller_nn, certificate_nn = train_controller_and_certificate(controller_nn, certificate_nn, data)
        #plot_function(certificate_nn, f"certificate_plot_{iteration}.png")
        print(f"Certificate plot saved as certificate_plot_{iteration}.png")
        #plot_function(controller_nn, f"controller_plot_after_{iteration}.png", type_wnnactor=True)
        print(f"Controller plot after fine-tuning saved as controller_plot_after_{iteration}.png")
        
        print("Translating the controller and certificate to DLG layers...")
        DLG_layers = DWN_to_logic_layers(certificate_nn)
        DLG_layers_controller = DWN_to_logic_layers(controller_nn.actor_mean)

        

        print("Encoding the verification formula in Z3...")
        #Temprature termometer encoding
        x_bits = [z3.Bool(f'x_{i}') for i in range(N_BITS)]
        y_bits = [z3.Bool(f'y_{i}') for i in range(N_BITS)]
        vx_bits = [z3.Bool(f'vx_{i}') for i in range(N_BITS)]
        vy_bits = [z3.Bool(f'vy_{i}') for i in range(N_BITS)]
        state_bits = [x_bits, y_bits, vx_bits, vy_bits]
        state_bits_flat = [x for row in state_bits for x in row]
        def get_thermometer_encoding_constraints(t_bits):
            t_larger_min = t_bits[0]  # t > min_temp if the first bit is 1
            return z3.And([t_larger_min] + [z3.Implies(t_bits[k], t_bits[k-1]) for k in range(1,N_BITS)])  # Theremometer encoding constraint for t
        
        states_constraints = z3.And([get_thermometer_encoding_constraints(bits) for bits in state_bits])
        #Barrier function encoding
        print("Encoding the barrier function constraints...")
        barrier_func_constraints, barrier_output_vars = encode_sat_model_with_vars_base_layers(DLG_layers, state_bits_flat, "h", "output")
        barrier_func_constraints = z3.And(barrier_func_constraints["constraints"])
        group_sum_barrier_map = get_groupsum_layer_output_mapping(certificate_nn, states=None)
        print("Encoding the mapping from group sum layer to the barrier output...")
        h_count = z3.Sum([z3.If(b, 1, 0) for b in barrier_output_vars])
        gs_barrier_constraint, h_real_output = group_sum_func_map_constraint(group_sum_barrier_map, "h_output_real", h_count)
        barrier_zero_idx = ((0 - BARRIER_MIN) // BARRIER_INTERVAL)
        barrier_constraint = z3.And(states_constraints, barrier_func_constraints, gs_barrier_constraint)
        
        print("Encoding initial set constraints")
        #Initial set constraints
        initial_set_constraints = initial_set_constraint(state_bits)
        #For checking h < 0, this is checked by not(h_bit) and  zero_bit; so that if any bit is 1 in the result then h < zero
        barrier_zero_bool_vec = [True if j <= barrier_zero_idx else False for j in range(BARRIER_DISC)]
        negative_barrier_constraints = negative_barrier_constraint(h_real_output, barrier_zero_bool_vec)
        initial_constraint = z3.And(initial_set_constraints, negative_barrier_constraints)

        print("Encoding unsafe set constraints")
        #unsafe set constraint
        unsafe_set_constraints = unsafe_set_constraint(state_bits)
        barrier_zero_bool_vec = [True if j < barrier_zero_idx else False for j in range(BARRIER_DISC)]
        unsafe_barrier_constraints = check_h_nonneg(h_real_output, barrier_zero_bool_vec)
        unsafe_constraint = z3.And(unsafe_set_constraints, unsafe_barrier_constraints)

        print("Encoding transition function and safety constraints")
        #((x′ = f (x, π(x)) ∧  h(x) >= 0) ∧ (h(x′) - h(x) < − γ.h(x))
        #The differnce between h and h' with h, has to be checked from the index which is 0
        #This can be achived by XORing h with zero
        #Checking h'> h can be done with or(h'and not h)
        #Case 1: h' > h then it is false
        #Case 2: h' < h then we have to check if the difference is less than h,:
        #   Do h and not zero_h
        #   Compare its count with the count(h not h') if the difference between h and h' is higher then it is true otherwise it is false 
        #Case 3: h = h' then false
        controller_func_constraints, controller_output_vars = encode_sat_model_with_vars_base_layers(DLG_layers_controller, state_bits_flat, "control", "output")
        controller_output_vars_x = controller_output_vars[:len(controller_output_vars)//2]
        controller_output_vars_y = controller_output_vars[len(controller_output_vars)//2:]
        controller_func_constraints = z3.And(controller_func_constraints["constraints"])
        group_sum_controller_map = get_groupsum_layer_output_mapping(controller_nn, type_wnnactor=True, states=None)
        c_count_x = z3.Sum([z3.If(b, 1, 0) for b in controller_output_vars_x])
        c_count_y = z3.Sum([z3.If(b, 1, 0) for b in controller_output_vars_y])
        control_real_output_constraint_x, control_real_output_x = group_sum_func_map_constraint(group_sum_controller_map["ux"], "control_output_real_x", c_count_x, CTRL_DISC)
        control_real_output_constraint_y, control_real_output_y = group_sum_func_map_constraint(group_sum_controller_map["uy"], "control_output_real_y", c_count_y, CTRL_DISC)

        print("Encoding the transition function constraints...")
        transition_constraint_formula, state_control_prime_vars, transition_inner_constraints = transiton_function_bitvector(state_bits, [control_real_output_x, control_real_output_y], "prime")
        print("Finished encoding the transition function constraints, encoding the safety constraints...")
        barrier_zero_bool_vec = [True if j < barrier_zero_idx else False for j in range(BARRIER_DISC)]
        nonneg_barrier_constraints = check_h_nonneg(h_real_output, barrier_zero_bool_vec)
        first_clause_constraints = z3.And(nonneg_barrier_constraints, transition_constraint_formula, z3.And(transition_inner_constraints), controller_func_constraints, control_real_output_constraint_x, control_real_output_constraint_y)

        print("Encoding the constraints for the second part of the last clause of the property...")
        #H' constraints
        state_prime_vars_flat = [x for row in state_control_prime_vars[:4] for x in row]
        barrier_prime_func_constraints, barrier_prime_output_vars = encode_sat_model_with_vars_base_layers(DLG_layers, state_prime_vars_flat, "h_prime", "output")
        barrier_prime_func_constraints = z3.And(barrier_prime_func_constraints["constraints"])
        h_prime_count = z3.Sum([z3.If(b, 1, 0) for b in barrier_prime_output_vars])
        gs_barrier_prime_constraint, h_prime_real_output = group_sum_func_map_constraint(group_sum_barrier_map, "h_prime_output_real", h_prime_count)
        prime_larger_h_constraint = z3.Or([z3.And(h_prime_real_output[i], z3.Not(h_real_output[i])) for i in range(BARRIER_DISC)])  # h' > h
        h_larger_prime_constraint = z3.Or([z3.And(h_real_output[i], z3.Not(h_prime_real_output[i])) for i in range(BARRIER_DISC)])  # h > h'
        h_diff_zero_count = z3.Sum([z3.If(z3.And(h_real_output[i], z3.Not(barrier_zero_bool_vec[i])),1,0) for i in range(BARRIER_DISC)])
        h_diff_prime_count = z3.Sum([z3.If(z3.And(h_real_output[i], z3.Not(h_prime_real_output[i])),1,0) for i in range(BARRIER_DISC)])
        safety_constraint = z3.If(prime_larger_h_constraint, z3.BoolVal(False), z3.If(h_larger_prime_constraint, h_diff_zero_count > h_diff_prime_count, z3.BoolVal(False)))
        second_clause_constraints = z3.And(barrier_prime_func_constraints, gs_barrier_prime_constraint, safety_constraint)
        transition_safety_constraint = z3.And(first_clause_constraints, second_clause_constraints)


        print("Start solving the constraints with Z3...")
        # Find a solution that satisfies the initial set constraint and negative barrier constraint
        solver = z3.Solver()
        solver.add(barrier_constraint)
        unsatisfiable = True
        for idx, clause in enumerate([initial_constraint, unsafe_constraint, transition_safety_constraint]):
            solver.push()  # Push the current state of the solver
            solver.add(clause)
            start_time = time.time()
            if solver.check() == z3.sat:
                unsatisfiable = False
                model = solver.model()
                end_time = time.time()
                print(f"Counterexample for clause {idx} found in {end_time - start_time:.2f} seconds")
                # Extract the counterexample
                x_values = [model.eval(state_bits[0][i]) for i in range(N_BITS)]
                y_values = [model.eval(state_bits[1][i]) for i in range(N_BITS)]
                vx_values = [model.eval(state_bits[2][i]) for i in range(N_BITS)]
                vy_values = [model.eval(state_bits[3][i]) for i in range(N_BITS)]
                #h_values = [model.eval(h_real_output[i]) for i in range(BARRIER_DISC)]
                
                # Convert thermometer encoding back to temperature value
                x_idx = sum(1 for v in x_values if v)
                y_idx = sum(1 for v in y_values if v)
                vx_idx = sum(1 for v in vx_values if v)
                vy_idx = sum(1 for v in vy_values if v)
                counterexample_state = []
                counterexample_x = -pos_limit + x_idx * POS_INTERVAL - POS_INTERVAL / 2.0
                counterexample_y = -pos_limit + y_idx * POS_INTERVAL - POS_INTERVAL / 2.0
                counterexample_vx = -vel_limit + vx_idx * VEL_INTERVAL - VEL_INTERVAL / 2.0
                counterexample_vy = -vel_limit + vy_idx * VEL_INTERVAL - VEL_INTERVAL / 2.0
                counterexample_state = [counterexample_x, counterexample_y, counterexample_vx, counterexample_vy]
                
                print(f"Counterexample found at state: {counterexample_state}")
                print(f"Barrier value h(x) should be negative")
                data.x_train.append(sample_points_around_counterexample(counterexample_state))
            else:
                end_time = time.time()
                print(f"No counterexample found in clause {idx} - property satisfied in {end_time - start_time:.2f} seconds")
            solver.pop()  # Pop the clause to reset the solver for the next clause
        

        if unsatisfiable:
            return controller_nn, certificate_nn
    
if __name__ == "__main__":      
    # Example usage
    trained_controller, trained_certificate = verification_loop(None, load_model=True)
    save_dir = os.path.join("/", "run1")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"AfterCEGIS_model.cleanrl_model")
    trained_controller.save_checkpoint(model_path, optimizer=None)
    certificate_path = os.path.join(save_dir, f"Certificate_model.certificate")
    torch.save(trained_certificate, certificate_path)