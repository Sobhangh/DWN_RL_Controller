# a function of the DWN layer
from typing import Literal

import torch
import torch.nn as nn
import torch_dwn as dwn
import torch.nn.functional as F

from difflogic_verify.utils.difflogic import BaseGroupSum, BaseLogicLayer, bin_op_vectorized, bin_op_vectorized_base, bin_op_vectorized_invert, gate_expression_16, gate_expression_6, gate_expression_not

def togroupsumLayer(self: dwn.GroupSum, in_dim, device):
    return BaseGroupSum(k=self.k, tau=self.tau, in_dim=in_dim, device=device)


def tobaseLogicLayer(self):
        """
        Converts the LUTLayer to a base logic layer (i.e. without learnable mapping).
        """
        #from difflogic_gpu.difflogic import BaseLogicLayer
        #print(self.mapping)
        #print(self.mapping.weights)
        #print("...")
        lgns = convert_to_logic_gate(self.luts.detach())
        # check if it is a leanrbale mapping
        if isinstance(self.mapping, dwn.LearnableMapping):
            #print("logic layer conversion with learnable mapping")
            selected_inputs = convert_mapping_to_selected_inputs(self.mapping.weights.detach().argmax(dim=0))
        else:
            # [[0,1],[2,3],...]
            selected_inputs = self.mapping.detach()
            
            #torch.arange(self.input_size).reshape(self.output_size, self.n)
            #print(selected_inputs)
            #print(selected_inputs.shape)
        #exit()
        # out = compute_logic_gate_output(x_og, selected_inputs, lgns)
        bbl = BaseLogicLayer(in_dim=int(self.input_size),
                       out_dim=int(self.output_size),
                       initalization='residual',
                       gate_set=16)
        #print(bbl.selected_inputs.shape)
        #print(selected_inputs.shape)
        bbl.selected_inputs[:,0] = selected_inputs[:,1]
        bbl.selected_inputs[:,1] = selected_inputs[:,0]
        with torch.no_grad():
            bbl.weights.copy_(lgns)
        return bbl


def convert_to_logic_gate(luts):
    """
    Convert LUTs (of shape (out_dim, 2**in_dim) for in_dim=2, shape=(out_dim, 4))
    into a one-hot encoding over the 16 possible 2-input logic gates.
    
    The canonical 16 2-input Boolean functions are defined (in the order below)
    as:
        op0  = [0,0,0,0]         (constant 0)
        op1  = [0,0,0,1]         (A and B)
        op2  = [0,0,1,0]         (A and not B, i.e. A and ~B)
        op3  = [0,0,1,1]         (A)
        op4  = [0,1,0,0]         (B and not A, i.e. B and ~A)
        op5  = [0,1,0,1]         (B)
        op6  = [0,1,1,0]         (A xor B)
        op7  = [0,1,1,1]         (A or B)
        op8  = [1,0,0,0]         (not(A or B))
        op9  = [1,0,0,1]         (A xnor B)
        op10 = [1,0,1,0]         (not B)
        op11 = [1,0,1,1]         (B implies A)
        op12 = [1,1,0,0]         (not A)
        op13 = [1,1,0,1]         (A implies B)
        op14 = [1,1,1,0]         (not(A and B))
        op15 = [1,1,1,1]         (constant 1)
    
    Parameters:
      luts   - a tensor of shape (out_dim, 4) containing the LUT values.
               (These are assumed to be continuous values initially, so we threshold them.)
      in_dim - the input dimension (for binary LUTs, in_dim should be 2).
    
    Returns:
      A tensor of shape (out_dim, 16) where each row is a one-hot (or indicator)
      vector over the 16 possible gate types.
    """
    out_dim = luts.shape[0]
    # Define the 16 canonical truth tables (each of length 4)
    canonical = torch.tensor([
        [0, 0, 0, 0],  # op0  constant 0
        [0, 0, 0, 1],  # op1  A and B
        [0, 0, 1, 0],  # op2  A and not B  (or equivalently: A - A*B)
        [0, 0, 1, 1],  # op3  A
        [0, 1, 0, 0],  # op4  B and not A  (or B - A*B)
        [0, 1, 0, 1],  # op5  B
        [0, 1, 1, 0],  # op6  A xor B      (A + B - 2*A*B)
        [0, 1, 1, 1],  # op7  A or B       (A + B - A*B)
        [1, 0, 0, 0],  # op8  not(A or B)
        [1, 0, 0, 1],  # op9  A xnor B     (not(A xor B))
        [1, 0, 1, 0],  # op10 not B
        [1, 0, 1, 1],  # op11 B implies A  (1 - B + A*B)
        [1, 1, 0, 0],  # op12 not A
        [1, 1, 0, 1],  # op13 A implies B  (1 - A + A*B)
        [1, 1, 1, 0],  # op14 not(A and B)
        [1, 1, 1, 1],  # op15 constant 1
    ], dtype=torch.int32, device=luts.device)  # shape (16,4)
    
    # Threshold the LUTs to get binary values (0 or 1)
    # Here we assume that values > 0 correspond to 1, otherwise 0.
    luts_bin = (luts > 0).to(torch.int32)  # shape: (out_dim, 4)
    
    # Expand dimensions to compare each LUT (out_dim x 4) against each canonical gate (16 x 4)
    # We want to get a tensor of shape (out_dim, 16, 4) where we compare each element.
    luts_bin_exp = luts_bin.unsqueeze(1)      # (out_dim, 1, 4)
    canonical_exp = canonical.unsqueeze(0)      # (1, 16, 4)
    
    # Compare: this produces a boolean tensor of shape (out_dim, 16, 4)
    eq = (luts_bin_exp == canonical_exp)
    
    # For each LUT and each candidate gate, check if all 4 entries match.
    match = eq.all(dim=2)  # shape: (out_dim, 16) with boolean values
    
    # Convert to float so that you have a one-hot style indicator (1.0 for a match, 0.0 otherwise)
    one_hot = match.to(torch.float32)
    
    # At this point, for each LUT we have a 16-dimensional vector indicating which gate(s)
    # match the LUT's truth table. For a properly hardened LUT, exactly one entry should be 1.
    #print(one_hot.shape)
    # count the number of 1s in each row
    #print(one_hot.sum(dim=0))
    return one_hot
def convert_mapping_to_selected_inputs(mapping):
    """
    Converts a 1D learnable mapping tensor into a 2D tensor of selected inputs.
    
    The function assumes that the input mapping is a 1D tensor of length 2*out_dim,
    where out_dim is the number of gates. Each consecutive pair of indices in the
    mapping is interpreted as the two input indices for a gate.
    
    Parameters:
      mapping (torch.Tensor): 1D tensor of indices, typically produced by weights.argmax(dim=0)
      
    Returns:
      torch.Tensor: A tensor of shape (out_dim, 2) where each row gives the two input
                    indices for the corresponding gate.
                    
    Raises:
      ValueError: If the length of mapping is not even.
    """
    # deepcopy of mapping
    # mapping = mapping.clone()
    # Flatten the mapping in case it is not already 1D
    mapping = mapping.flatten()
    num_elements = mapping.numel()
    
    if num_elements % 2 != 0:
        raise ValueError("Mapping length must be even to form pairs for each gate.")
    
    # Calculate the number of gates (out_dim)
    out_dim = num_elements // 2
    
    # Reshape into (out_dim, 2)
    selected_inputs = torch.zeros((out_dim,2))#mapping.view(out_dim, 2)
    selected_inputs[:, 0] = mapping[0::2]  # First input index for each gate
    selected_inputs[:, 1] = mapping[1::2]  # Second input index for each gate
    # print(selected_inputs)
    return selected_inputs.long()




# class BaseLogicLayer(nn.Module):
#     """
#     A hardened version of a scalable differentiable logic gate network layer with fixed connections and gates.
    
#     In this version, the forward pass is hardened: a single connection per output is selected
#     (using a fixed index) rather than a soft probability distribution. The gate weights are learned as before.
#     """
#     def __init__(
#             self,
#             in_dim: int,
#             out_dim: int,
#             device: str = 'cpu',
#             grad_factor: float = 1.,
#             initalization: str = 'random',
#             seed = None,
#             gate_set : int = Literal[2,6,16],  #choices between 6 and 16 (2 is inverter Layer) 
#             **kwargs
#     ):
#         """
#         :param in_dim:         Input dimensionality of the layer.
#         :param out_dim:        Output dimensionality of the layer.
#         :param num_connections:Number of random input connections per output.
#         :param device:         Device (e.g., 'cuda' or 'cpu').
#         :param grad_factor:    Gradient factor.
#         :param hardened_index: Fixed connection index (0 <= index < num_connections) to use for every output.
#         :param initalization:  Initialization method for weights ('random' or 'residual').
#         :param seed:           Optional random seed.
#         :param extra_not:      If True, additional NOT weights are created.
#         :param disable_not:    If True, the extra NOT branch is disabled.
#         :param freeze_interconnect: Unused in hardened mode.
#         """
#         super().__init__()
#         if seed is not None:
#             torch.manual_seed(seed)
        
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.device = device
#         self.grad_factor = grad_factor
#         if gate_set == 6:
#             self.bin_op = bin_op_vectorized_base
#         elif gate_set == 16:
#             self.bin_op = bin_op_vectorized
#         elif gate_set == 2:
#             self.bin_op = bin_op_vectorized_invert
#         self.gate_set = gate_set
#         # Create the selected_inputs buffer.
#         if self.in_dim > 1:
#             selected = torch.stack([
#                 torch.randperm(self.in_dim, device=device)[:2]
#                 for _ in range(self.out_dim)
#             ])
#         else:
#             selected = torch.zeros((self.out_dim, 2), device=device, dtype=torch.long)
#         # for intepretability
#         c = torch.randperm(2 * self.out_dim) % self.in_dim
#         c = torch.randperm(self.in_dim)[c]
#         c = c.reshape(self.out_dim, 2)
#         selected[:, :2] = c
#         self.register_buffer('selected_inputs', selected)
        
#         self.weights_size = gate_set
#         if initalization == 'random':
#             init_tensor = torch.rand(out_dim, self.weights_size, device=device)
#             for i in range(out_dim):
#                 rand_idx = torch.randint(0, self.weights_size, (1,)).item()
#                 init_tensor[i, rand_idx] *= 5.0
#             self.weights = nn.Parameter(init_tensor)
        
#         elif initalization == 'residual':
#             weight_on_a = 2.0
#             weight_on_rest = 1.0
#             weight_vec = torch.ones(out_dim, self.weights_size, device=device) * weight_on_rest
#             weight_vec[:, 3] = weight_on_a
#             weight_vec = weight_vec / weight_vec.sum(dim=-1, keepdim=True)
#             self.weights = nn.Parameter(weight_vec)
#         else:
#             raise ValueError("Only 'random' and 'residual' initalization is supported.")
        
#         self.num_neurons = out_dim
#         self.num_weights = out_dim
#     def extra_repr(self):
#         return f"in_dim={self.in_dim}, out_dim={self.out_dim}, gate_set={self.weights_size}"

#     def forward(self, x):
#         """
#         Hardened forward pass: uses fixed (hardened) connection indices.
#         Instead of computing a soft distribution over candidate connections, this function
#         directly selects one connection per output using the fixed hardened_index.
        
#         Args:
#             x (torch.Tensor): Input tensor of shape [batch_size, in_dim].
        
#         Returns:
#             torch.Tensor: Output tensor after applying the binary logic operation.
#         """
#         # x: [batch_size, in_dim]
#         batch_size = x.size(0)
#         #print(batch_size)
#         # For each output row, use the fixed hardened_index to select a single input connection.
#         row_indices = torch.arange(self.out_dim, device=self.device)
#         # selected_inputs has shape [out_dim, num_connections]
#         # Use the hardened index for both a and b.
#         #print(self.selected_inputs.shape)
#         #print(np.unique(self.selected_inputs.cpu().numpy()))
#         indices_a = self.selected_inputs[:, 0]
#         indices_b = self.selected_inputs[:, 1]
        
#         # Gather the corresponding input values.
#         # a and b will be [batch_size, out_dim]
#         # to numpy selected input and print uniques
#         #print(indices_a)
#         a = x[:, indices_a]
#         b = x[:, indices_b]
#         # Compute gate weights.
#         assert a.shape == b.shape == (batch_size, self.weights.shape[0]), (a.shape, b.shape, self.weights.shape)
#         gate_weights = F.one_hot(self.weights.argmax(dim=-1), num_classes=self.weights_size).float()
#         #print(gate_weights.shape)
#         output = self.bin_op(a, b, gate_weights)
#         #print(output.shape)
#         #print("retunred")
#         return output
#     def forward_z3_logic(self, input_z3_expressions):
#         """
#         Given a list (or array) of Z3 expressions for the inputs (of length self.in_dim),
#         compute the output Z3 expression for each gate in this layer.
#         For each gate, use the fixed connection indices from selected_inputs:
#             A_expr = input_z3_expressions[selected_inputs[i, 0]]
#             B_expr = input_z3_expressions[selected_inputs[i, 1]]
#         Then, using the gate's operator (given by the argmax of weights[i]) and the appropriate gate_expression
#         function (based on self.gate_set), compute the gate's Z3 expression.
#         Returns a list of Z3 expressions (of length self.out_dim).
#         """
#         # Extract indices (as numpy arrays)
#         a_indices = self.selected_inputs[:, 0].cpu().numpy()
#         b_indices = self.selected_inputs[:, 1].cpu().numpy()
#         op_ids = self.weights.argmax(dim=-1).cpu().numpy().astype(int)
        
#         # Choose the proper gate_expression function.
#         if self.gate_set == 2:
#             gate_expression_func = gate_expression_not
#         elif self.gate_set == 16:
#             gate_expression_func = gate_expression_16
#         elif self.gate_set == 6:
#             gate_expression_func = gate_expression_6
#         else:
#             raise ValueError(f"Invalid gate_set: {self.gate_set}")
        
#         output_exprs = []
#         for i in range(len(a_indices)):
#             A_expr = input_z3_expressions[a_indices[i]]
#             B_expr = input_z3_expressions[b_indices[i]]
#             op_id = op_ids[i]
#             out_expr = gate_expression_func(op_id, A_expr, B_expr)
#             output_exprs.append(out_expr)
#         return output_exprs

def DWN_to_logic_layers(dwn_model):
    logx_layers  = []
    idx = 0
    for layer in dwn_model.net:
        if isinstance(layer, dwn.LUTLayer):
            logx_layers.append(tobaseLogicLayer(layer))
        if isinstance(layer, dwn.GroupSum):
            logx_layers.append(togroupsumLayer(layer, in_dim=dwn_model.net[idx-1].output_size, device=dwn_model.device))
        idx += 1
    return logx_layers