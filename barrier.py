import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import tqdm
import z3
from wnn_models import WNN
import torch
import torch.nn.functional as F
from conversion import DWN_to_logic_layers
from difflogic_verify.utils.verification_utils import encode_sat_model_with_vars_base_layers
from CtrlRoomTemp import *
import time


"""
Training Formulas and theoretical aspects:
S := {x(k) ∈ D | h(x(k)) ≥ 0}
 control input uk : S → U, ∀xk ∈ S and γ > 0 such
that
 − γ.h(xk) ≤ ∆h (xk, uk). 
where ∆h (xk, uk) = h(f(xk, uk)) − h(xk) is the change in the barrier function along the system trajectories.
Domain of h(x) can be considered from a small negative value to a "medium" positive value e.g. -1 to 9 

The loss function for training the controller and certificate is given by:
L = (1/N_I) * SUM(m(-h(d))| d∈D_I) + (1/N_U) * SUM(m(h(d))| d∈D_U) + (1/N) * SUM(m(-∆h (xk, uk) - γ.h(xk))| d∈Internal(S))

Note that this function penalises points in D□ where the required condition is not satisfied. Suitable choices for
function m are leaky-ReLU, which is piecewise linear, and softplus, which is smooth. Since several of the sets
X□ are boundaries of sets, or represent level sets, in practice we consider a small band around them, in order
to encompass a sufficient number of data points in D□.

TO DO:
Special term for controller?
δ1 > 0 and δ2 > 0, from the satellite paper for the relu loss function used for overapproximation?



VERIFICATION FORMULA TO CHECK:
    
Negation of the property which was previously in the form of loss function: 
∃x ∈ X :(x ∈ XI ∧ h(x) < 0) ∨ (x ∈ XU ∧ h(x) ≥ 0) ∨ ((x′ = f (x, π(x)) ∧  h(x) >= 0) ∧ (h(x′) - h(x) < − γ.h(x)) )

Remarks:
There can be problems in checking the boundries of initial and unsafe sets: Overaproximate both if the boundry interval is not completely inside it?
For the last clause, if gamma is set to 1, then it can be checked by first checking if  h(x′) >= h(x), in that case no further check is necessary, if not then 
substract the two and check how it compares to h(x);
"""

# N_SAMPLE_I = 1000
# N_SAMPLE_U = 1000
# N_SAMPLE_REST = 1000
N_SAMPLE_TOTAL = 10000
BARRIER_DISC = 100


def sample_points(domain):
    """
    Sample points from the given domain. The domain can be the safe set, unsafe set or the rest of the state space. The function returns a list of sampled points.
    """
    return np.random.uniform(domain[0], domain[1], size=(N_SAMPLE_TOTAL, 1)).astype(np.float32)


def plot_function(funct, save_path: str = "certificate_plot.png", type_wnnactor: bool = False):
    x_vals = np.linspace(MIN_TEMP, MAX_TEMP, 200)
    x_tensor = torch.from_numpy(x_vals.reshape(-1, 1)).float()
    device = next(funct.parameters()).device
    x_tensor = x_tensor.to(device)

    funct.eval()
    with torch.no_grad():
        if type_wnnactor:
            _, _, f_vals = funct.get_action(x_tensor)
        else:
            f_vals = funct(x_tensor)
        #print(f"fbals {f_vals}")
        f_vals = f_vals.cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, f_vals, 'b-', linewidth=2)
    
    plt.axvline(x=CtrlRoomTemp.INITIAL[0], color='g', linestyle='--', alpha=0.5, label='Initial set')
    plt.axvline(x=CtrlRoomTemp.INITIAL[1], color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=CtrlRoomTemp.VALID[0], color='orange', linestyle='--', alpha=0.5, label='Valid set')
    plt.axvline(x=CtrlRoomTemp.VALID[1], color='orange', linestyle='--', alpha=0.5)

    if not type_wnnactor:
        plt.axhline(y=0, color='r', linestyle='--', label='h=0 (barrier boundary)')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Certificate h(x)')
        plt.title('Barrier Certificate Function')
    else:
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Controller output u(x)')
        plt.title('Controller Function')

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    #plt.show()

def train_controller_and_certificate(controller_nn, certificate_nn: WNN, points):
    """
    Function for training the controller and certificate. The function samples points from the safe set, unsafe set and the rest of the state space and uses them to compute the loss function for training the controller and certificate. The function returns the trained controller and certificate.
    """
    # Sample points from the safe set, unsafe set and the rest of the state space
    #points = sample_points((15.0, 32.0))
    
    device = "cuda" #next(controller_nn.parameters()).device
    # controller_nn = controller_nn.to(device)
    # certificate_nn = certificate_nn.to(device)

    # # Thermometer is not an nn.Module, so keep its thresholds manually in sync.
    # if hasattr(controller_nn, "thermometer") and getattr(controller_nn.thermometer, "thresholds", None) is not None:
    #     controller_nn.thermometer.thresholds = controller_nn.thermometer.thresholds.to(device)
    # if hasattr(certificate_nn, "thermometer") and getattr(certificate_nn.thermometer, "thresholds", None) is not None:
    #     certificate_nn.thermometer.thresholds = certificate_nn.thermometer.thresholds.to(device)
    print("starting fine-tuning of the controller and certificate")
    controller_nn.train()
    certificate_nn.train()

    # Hyperparameters
    epochs = 300
    batch_size = 256
    lr = 1e-2
    gamma = 1

    x_all = torch.tensor(points, device=device, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(
         list(certificate_nn.parameters()) + list(controller_nn.parameters()), #+ list(controller_nn.parameters())
        lr=lr,
    )

    history = []

    # Train until the loss is (numerically) zero
    max_epochs = 5000
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
            initial_mask = (xb >= CtrlRoomTemp.INITIAL[0]) & (xb <= CtrlRoomTemp.INITIAL[1])
            safe_mask = (xb >= CtrlRoomTemp.VALID[0]) & (xb <= CtrlRoomTemp.VALID[1])
            unsafe_mask = ~safe_mask
            # Controller output constrained to [0, 1]
            #TO DO: How to work with a thermoter output? how can it be in training?
            #Possible solution: use the normal multi class classification and have it be converted to thermometer encoding in the verification step?
            #u is between 0 and 1 and is classfied into CTRL_DISC classes
            #h is between -1 and 9 and is classified into BARRIER_DISC classes
            # prob_u = controller_nn(xb)  # (B, 1)
            # prob_h_x = certificate_nn(xb)
            # for hidx in range(prob_h_x.shape[1]):
            #     h_x = torch.full((xb.shape[0], 1), hidx * BARRIER_INTERVAL + BARRIER_INTERVAL / 2.0, device=device, dtype=torch.float32)
            #     interior_h = (h_x > 0) 
            #     for i in range(prob_u.shape[1]):
            #         #Use the mean of u in the inteval
            #         u = torch.full((xb.shape[0], 1), i * CTRL_INTERVAL + CTRL_INTERVAL / 2.0, device=device, dtype=torch.float32)

            #         # Barrier values h(x) and h(f(x,u))
            #         x_next = xb + tau * (alpha_e * (temp_e - xb) + alpha_h * (temp_h - xb) * u)
            #         #TO DO: Same for h_next as well?
            #         prob_h_next = certificate_nn(x_next)
            #         for hidx_next in range(prob_h_next.shape[1]):
            #             h_next = torch.full((xb.shape[0], 1), hidx_next * BARRIER_INTERVAL + BARRIER_INTERVAL / 2.0, device=device, dtype=torch.float32)
            #             delta_h = h_next - h_x
            #             # interior_h.sum() is equal to xb.shape[0] if h_x > 0 and otherwise it is all False which loss_d is zero
            #             #same for prob_u[interior_h, i] and other probabilities they can also be replaced with prob_u[:, i]
            #             loss_d = (F.relu(-delta_h - gamma * h_x)[interior_h] * prob_u[interior_h, i] * prob_h_next[interior_h, hidx_next] * prob_h_x[interior_h, hidx]).sum() / interior_h.sum() if interior_h.any() else zero
            #             loss +=  loss_d

            #     loss_i = (F.relu(-h_x[initial_mask]) * prob_h_x[initial_mask, hidx]).sum() / initial_mask.sum() if initial_mask.any() else zero
            #     loss_u = (F.relu(h_x[unsafe_mask]) * prob_h_x[unsafe_mask, hidx]).sum() / unsafe_mask.sum() if unsafe_mask.any() else zero
            #     loss += loss_i + loss_u

            #Simple version for the case the output of controller and certificate is "real" due to the regression head
            # Compute the loss function for training the controller and certificate
            # L = (1/N_I) * SUM(m(-h(d))| d∈D_I) + (1/N_U) * SUM(m(h(d))| d∈D_U) + (1/N) * SUM(m(-∆h (xk, uk) - γ.h(xk))| d∈Internal(S))
            _,_,u = controller_nn.get_action(xb)  # (B, 1)
            h_x = certificate_nn(xb)  # (B, 1)
            interior_h = (h_x > 0) 
            x_next = xb + tau * (alpha_e * (temp_e - xb) + alpha_h * (temp_h - xb) * u)
            h_next = certificate_nn(x_next)  # (B, 1)
            delta_h = h_next - h_x
            loss_i = F.relu(-h_x[initial_mask]).sum() / initial_mask.sum() if initial_mask.any() else zero
            loss_u = F.relu(h_x[unsafe_mask]).sum() / unsafe_mask.sum() if unsafe_mask.any() else zero
            loss_d = F.relu(-delta_h - gamma * h_x)[interior_h].sum() / interior_h.sum() if interior_h.any() else zero
            loss += loss_i + loss_u + loss_d
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     list(controller_nn.parameters()) + list(certificate_nn.parameters()), 1.0
            # )
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

        history.append(epoch_loss)

        # Stop when loss is effectively zero (or if max epochs reached)
        if epoch_loss <= tol: #or epoch_loss <= tol
            print(f"Stopping training at epoch {epoch} with loss {epoch_loss:.4f}")
            break
    print(f"Final Epoch {epoch}, Loss: {epoch_loss:.4f}")
    print("finished fine-tuning of the controller and certificate")
    return controller_nn, certificate_nn


def lower_upper_transition_function():
    #Find the minimum and maximum of the transition function in each discretized interval of x and u to get the lower and upper bounds for the transition function in each interval. 
    #This is done by finding the critical points of the transition function and checking the values at the critical points and the boundaries of the intervals.
    x, u = sp.symbols('x u', real=True)
    f = x + tau * (alpha_e * (temp_e - x) + alpha_h * (temp_h - x) * u)

    fx = sp.diff(f, x)
    fu = sp.diff(f, u)

    critical_points = sp.solve([fx, fu], [x, u], dict=True)
    print(critical_points)

    def get_disc_index(value, min_value, interval):
        if value < min_value:
            return 0
        return int((value - min_value) / interval)

    lower_transition = {}
    upper_transition = {}
    for i in range(TEMP_DISC) :
        x_start = MIN_TEMP + i * TEMP_INTERVAL
        x_end = x_start + TEMP_INTERVAL
        for j in range(CTRL_DISC) :
            u_start = j * CTRL_INTERVAL
            u_end = u_start + CTRL_INTERVAL
            cps = [cp for cp in critical_points if cp[x] > x_start and cp[x] < x_end and cp[u] > u_start and cp[u] < u_end]
            cps.append({x: x_start, u: u_start})
            cps.append({x: x_start, u: u_end})
            cps.append({x: x_end, u: u_start})
            cps.append({x: x_end, u: u_end})
            transitions = [transition(cp[x], cp[u]) for cp in cps]
            lower_transition[(i, j)] =  min(get_disc_index(min(transitions), MIN_TEMP, TEMP_INTERVAL), TEMP_DISC-1)
            upper_transition[(i, j)] = min(get_disc_index(max(transitions), MIN_TEMP, TEMP_INTERVAL), TEMP_DISC-1)
    return lower_transition, upper_transition

def thermometer_encode_transition(lower_transition, upper_transition, x_bits, u_bits, name):
    """
    Function for generating Z3 expressions for the transition function. It is used in the verification step to encode the transition function as a Z3 expression.
    It returns 2 Z3 expressions, one for the lower bound and one for the upper bound of the transition function. 
    The input and output are encoded as thermometer bits. For each thermometer bit, there is a Z3 boolean variable.
    The outputs should be in the form of if thermometer_encoded_x == value_x and thermometer_encoded_u == value_u then thermometer_encoded_output = value_output, where value_x, value_u and value_output are the corresponding indices of the thermometer bits in the precomputed dictionaries for lower and upper bounds.
    """
    lower_out_bits = [z3.Bool(f'lower_out_{i}') for i in range(TEMP_DISC)]
    upper_out_bits = [z3.Bool(f'upper_out_{i}') for i in range(TEMP_DISC)]
    x_prime_bits = [z3.Bool(f'{name}_{i}') for i in range(TEMP_DISC)]

    # Build constraints for each (x, u) discretized pair
    constraints_lower = []
    constraints_upper = []
    x_match_dict = {}
    for i in range(TEMP_DISC):
        x_match_dict[i] = z3.And([x_bits[k] == (k <= i) for k in range(TEMP_DISC)])
    u_match_dict = {}
    for j in range(CTRL_DISC):
        u_match_dict[j] = z3.And([u_bits[k] == (k <= j) for k in range(CTRL_DISC)])
    lower_out_dict = {}
    upper_out_dict = {}
    for i in range(TEMP_DISC):
        lower_out_dict[i] = z3.And([lower_out_bits[k] == (k <= i) for k in range(TEMP_DISC)])
        upper_out_dict[i] = z3.And([upper_out_bits[k] == (k <= i) for k in range(TEMP_DISC)])


    for i in range(TEMP_DISC):
        for j in range(CTRL_DISC):
            # Condition: x_bits match index i AND u_bits match index j
            x_match = x_match_dict[i]
            u_match = u_match_dict[j]
            
            # Implication: if input matches, then output matches precomputed bounds
            lower_idx = lower_transition[(i, j)]
            upper_idx = upper_transition[(i, j)]
            
            # lower_out = z3.And([lower_out_bits[k] == (k <= lower_idx) for k in range(TEMP_DISC)])
            # upper_out = z3.And([upper_out_bits[k] == (k <= upper_idx) for k in range(TEMP_DISC)])
            lower_out = lower_out_dict[lower_idx]
            upper_out = upper_out_dict[upper_idx]
            
            constraints_lower.append(z3.Implies(z3.And(x_match, u_match), lower_out))
            constraints_upper.append(z3.Implies(z3.And(x_match, u_match), upper_out))
    
    x_prime_constraints = [z3.Implies(x_prime_bits[k], x_prime_bits[k-1]) for k in range(1,TEMP_DISC)]  # Theremometer encoding constraint for x'
    x_prime_constraints += [z3.Implies(z3.Not(upper_out_bits[k]), z3.Not(x_prime_bits[k])) for k in range(TEMP_DISC)]  #  x' should be less than or equal to the upper bound
    x_prime_constraints += [z3.Implies(lower_out_bits[k], x_prime_bits[k]) for k in range(TEMP_DISC)]  # x' should be greater than or equal to the lower bound 

    return z3.And(constraints_lower + constraints_upper + x_prime_constraints), x_prime_bits

#Initializeing bounds for barrier: These are just initial values, the actual bounds can be inferred from the alpha and beta parameters in the regression head of the certificate neural network after training; the number of intervals for each of the spaces can be a hyperprameter;
BARRIER_MAX = 3
BARRIER_MIN = -1
BARRIER_INTERVAL = (BARRIER_MAX - BARRIER_MIN) / BARRIER_DISC
def get_groupsum_layer_output_mapping(model, type_wnnactor: bool = False):
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()

        return hook
    # Register a forward hook on the layer before the groupsum layer
    if not type_wnnactor:
        model.net[-2].register_forward_hook(get_features("group_sum"))
    else:
        model.fc_mean.net[-2].register_forward_hook(get_features("group_sum"))
    x = []
    for i in range(TEMP_DISC):
        temp = MIN_TEMP + i * TEMP_INTERVAL + TEMP_INTERVAL / 2.0
        x.append(temp)
        
    x = np.array(x, dtype=np.float32)
    x = torch.from_numpy(x.reshape(-1, 1)).float()
    device = next(model.parameters()).device
    x = x.to(device)
    if type_wnnactor:
        #print(f"input to the model, size {x.shape}")
        _,_,y = model.get_action(x)
    else:
        y = model(x)
    grps_y = features["group_sum"]
    #The bounds for controller it is beween 0 and 1; for certificate
    #it has to be infered from the alpha and beta parameter in the regression head; the number of intervals for each of the spaces can be a hyperprameter; 
    # BARRIER_DISC for certificate and CTRL_DISC for controller; 
    group_sum_func_map = {}
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
    for i, gs in enumerate(grps_y.detach().cpu()):
        group_sum_func_map[gs] = int((y[i].cpu() - BARRIER_MIN) // BARRIER_INTERVAL) if not type_wnnactor else int(y[i].cpu() // CTRL_INTERVAL)
    

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

def initial_set_constraint(t_bits):
    start_idx = int((CtrlRoomTemp.INITIAL[0] - MIN_TEMP) / TEMP_INTERVAL)
    end_idx = int((CtrlRoomTemp.INITIAL[1] - MIN_TEMP) / TEMP_INTERVAL)
    constraints = [z3.And([t_bits[i] for i in range(start_idx+1)])] # t_bits should be greater than or equal to the start index 
    constraints += [z3.And([z3.Not(t_bits[i]) for i in range(end_idx+1, TEMP_DISC)])]  # t_bits should be less than or equal to the end index
    return z3.And(constraints)

def unsafe_set_constraint(t_bits):
    start_idx = int((CtrlRoomTemp.VALID[0] - MIN_TEMP) / TEMP_INTERVAL)
    end_idx = int((CtrlRoomTemp.VALID[1] - MIN_TEMP) / TEMP_INTERVAL)
    #Overapproximating unsafe set by including the boundary
    constraints = [z3.And([t_bits[i] for i in range(start_idx)])]  # t_bits should be greater than or equal to the start index 
    constraints += [z3.And([z3.Not(t_bits[i]) for i in range(end_idx, TEMP_DISC)])]  # t_bits should be less than or equal to the end index
    return z3.Not(z3.Or(constraints))

# sorted_bool_vecs = []
# for i in range(TEMP_DISC+1):
#     sorted_bool_vecs.append([True if j<i else False for j in range(TEMP_DISC)])

# def get_sorted_output(unsorted_bool_list, name):
#     sorted_output = [z3.Bool(f'{name}_{i}') for i in range(TEMP_DISC)]
#     #Is it possible to introduce an integer variable for count of the unsorted list and just compare k to it?
#     def rec_cond(k): 
#         if k == 0:
#             return z3.And([sorted_output[i] == False for i in range(TEMP_DISC)])
#         return z3.If(z3.Exactly(unsorted_bool_list, k), z3.And([sorted_output[i] == sorted_bool_vecs[k][i] for i in range(TEMP_DISC)]), rec_cond(k-1))
#     return rec_cond(TEMP_DISC)

def sample_points_around_counterexample(counterexample):
    """
    Function for sampling points around a counterexample. This can be used to find more counterexamples in the neighborhood of the initial counterexample returned by the solver, which can help in better understanding the failure cases and improving the training of the controller and certificate.
    The function takes in a counterexample point, a radius for sampling around that point, and the number of points to sample. It returns a list of sampled points around the counterexample.
    """
    radius = (MAX_TEMP - MIN_TEMP) / 15  
    num_points = N_SAMPLE_TOTAL // 10
    return np.random.normal(counterexample, radius , size=(num_points, 1)).astype(np.float32)

"""
Composing the verification formula:
∃x ∈ X :(x ∈ XI ∧ h(x) < 0) ∨ (x ∈ XU ∧ h(x) ≥ 0) ∨ ((x′ = f (x, π(x)) ∧  h(x) >= 0) ∧ (h(x′) - h(x) < − γ.h(x)) )

TO DO:
Doing Or in parallel
Multiple counterexamples returned by z3?
points around counterexample from Fossil
"""

def verification_loop(controller_nn, certificate_nn: WNN):
    points = sample_points((MIN_TEMP, MAX_TEMP))
    lower_bound, upper_bound = lower_upper_transition_function()

    iteration = -1
    while True:
        iteration += 1
        print(f"Starting verification iteration {iteration}...")
        controller_nn, certificate_nn = train_controller_and_certificate(controller_nn, certificate_nn, points)
        plot_function(certificate_nn, f"certificate_plot_{iteration}.png")
        print(f"Certificate plot saved as certificate_plot_{iteration}.png")
        plot_function(controller_nn, f"controller_plot_after_{iteration}.png", type_wnnactor=True)
        print(f"Controller plot after fine-tuning saved as controller_plot_after_{iteration}.png")
        
        print("Translating the controller and certificate to DLG layers...")
        DLG_layers = DWN_to_logic_layers(certificate_nn)
        DLG_layers_controller = DWN_to_logic_layers(controller_nn.fc_mean)

        print("Encoding the verification formula in Z3...")
        #Temprature termometer encoding
        t_bits = [z3.Bool(f't_{i}') for i in range(TEMP_DISC)]
        t_larger_min = t_bits[0]  # t > min_temp if the first bit is 1
        t_constraints = z3.And([t_larger_min] + [z3.Implies(t_bits[k], t_bits[k-1]) for k in range(1,TEMP_DISC)])  # Theremometer encoding constraint for t
        
        #Barrier function encoding
        barrier_func_constraints, barrier_output_vars = encode_sat_model_with_vars_base_layers(DLG_layers, t_bits, "h", "output")
        barrier_func_constraints = z3.And(barrier_func_constraints["constraints"])
        group_sum_barrier_map = get_groupsum_layer_output_mapping(certificate_nn)
        h_count = z3.Sum([z3.If(b, 1, 0) for b in barrier_output_vars])
        gs_barrier_constraint, h_real_output = group_sum_func_map_constraint(group_sum_barrier_map, "h_output_real", h_count)
        barrier_zero_idx = ((0 - BARRIER_MIN) // BARRIER_INTERVAL)
        barrier_constraint = z3.And(t_constraints, barrier_func_constraints, gs_barrier_constraint)
        
        print("Encoding initial set constraints")
        #Initial set constraints
        initial_set_constraints = initial_set_constraint(t_bits)
        #For checking h < 0, this is checked by not(h_bit) and  zero_bit; so that if any bit is 1 in the result then h < zero
        barrier_zero_bool_vec = [True if j <= barrier_zero_idx else False for j in range(BARRIER_DISC)]
        negative_barrier_constraints = negative_barrier_constraint(h_real_output, barrier_zero_bool_vec)
        initial_constraint = z3.And(initial_set_constraints, negative_barrier_constraints)

        print("Encoding unsafe set constraints")
        #unsafe set constraint
        unsafe_set_constraints = unsafe_set_constraint(t_bits)
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
        controller_func_constraints, controller_output_vars = encode_sat_model_with_vars_base_layers(DLG_layers_controller, t_bits, "control", "output")
        controller_func_constraints = z3.And(controller_func_constraints["constraints"])
        group_sum_controller_map = get_groupsum_layer_output_mapping(controller_nn, type_wnnactor=True)
        c_count = z3.Sum([z3.If(b, 1, 0) for b in controller_output_vars])
        control_real_output_constraint, control_real_output = group_sum_func_map_constraint(group_sum_controller_map, "control_output_real", c_count, CTRL_DISC)

        print("Encoding the transition function constraints...")
        transition_constraints, t_prime_vars = thermometer_encode_transition(lower_bound, upper_bound, t_bits, control_real_output, "t_prime")
        print("Finished encoding the transition function constraints, encoding the safety constraints...")
        barrier_zero_bool_vec = [True if j < barrier_zero_idx else False for j in range(BARRIER_DISC)]
        nonneg_barrier_constraints = check_h_nonneg(h_real_output, barrier_zero_bool_vec)
        first_clause_constraints = z3.And(nonneg_barrier_constraints, transition_constraints, controller_func_constraints, control_real_output_constraint)

        print("Encoding the constraints for the second part of the last clause of the property...")
        #H' constraints
        barrier_prime_func_constraints, barrier_prime_output_vars = encode_sat_model_with_vars_base_layers(DLG_layers, t_prime_vars, "h_prime", "output")
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
                t_values = [model.eval(t_bits[i]) for i in range(TEMP_DISC)]
                #h_values = [model.eval(h_real_output[i]) for i in range(BARRIER_DISC)]
                
                # Convert thermometer encoding back to temperature value
                temp_idx = sum(1 for v in t_values if v)
                counterexample_temp = MIN_TEMP + temp_idx * TEMP_INTERVAL - TEMP_INTERVAL / 2.0
                
                print(f"Counterexample found at temperature: {counterexample_temp:.4f}")
                print(f"Barrier value h(x) should be negative")
                points.append(sample_points_around_counterexample(counterexample_temp))
            else:
                end_time = time.time()
                print(f"No counterexample found in clause {idx} - property satisfied in {end_time - start_time:.2f} seconds")
            solver.pop()  # Pop the clause to reset the solver for the next clause
        

        if unsatisfiable:
            return controller_nn, certificate_nn
    

