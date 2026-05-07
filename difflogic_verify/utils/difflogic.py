"""
This is an adaption of the code from Petersen et al. (2022) Deep Differentiable Logic Gate Networks
Check out the original code at: https://github.com/Felix-Petersen/difflogic
This version, does not use cuda but a torch version. Making it straightforward to run, install and modify. 

The second difference is that connections are saved. Also GroupSum conenctions are not assumed to be fixed.
"""

import torch
import numpy as np
from typing import Literal

########################################################################################################################
from torch.nn import functional as F

import torch
import torch.nn as nn
import numpy as np


@torch.jit.script
def bin_op_vectorized_base(a, b, gate_weights):
    """
    Compute 8 binary operations in parallel, then for each operation
    decide (via not_weights) whether to invert the operation's output.

    Args:
        a, b: Tensors of shape (batch, out_dim).
        gate_weights: Tensor of shape (out_dim, 8) representing the mixture
            weights for the 8 operations.
        not_weights: Tensor of shape (out_dim, 2) representing the mixing
            weights for choosing between the base op and its inversion.
            (E.g. via softmax, where index 0 corresponds to not-inverting and
            index 1 to inverting.)

    Returns:
        result: Tensor of shape (batch, out_dim), computed as a weighted sum
            of the effective operations.
    """
    # Precompute common subexpressions
    ab = a * b
    ab2 = 2 * ab

    # Define 8 base operations (each will later be paired with its inverse).
    # The chosen operations are one representative from each complementary pair:
    op0 = torch.zeros_like(a)  # 0; inversion yields 1.
    op1 = ab  # A and B; inversion: 1 - (a*b)
    op2 = a - ab  # not(A implies B); inversion: 1 - a + ab
    op3 = a  # A; inversion: 1 - a
    # op4 = b - ab                 # not(B implies A); inversion: 1 - b + ab
    # op5 = b                      # B; inversion: 1 - b
    op6 = a + b - ab2  # A xor B; inversion: 1 - a - b + 2ab
    op7 = a + b - ab  # A or B; inversion: 1 - a - b + ab

    # Stack operations along a new dimension: shape (batch, out_dim, 8)
    ops = torch.stack([op0, op1, op2, op3, op6, op7], dim=-1)

    # Compute the inverted version of each op: simply 1 - op.
    effective_ops = ops
    result = torch.einsum("bod,od->bo", effective_ops, gate_weights)
    # Expand gate_weights (shape (out_dim, 8)) to (1, out_dim, 8)
    # gate_weights = gate_weights.unsqueeze(0)

    # Mix the 8 effective operations per neuron.
    # result = torch.sum(effective_ops * gate_weights, dim=-1)  # shape (batch, out_dim)

    # print(torch.allclose(result, result_t))
    return result


@torch.jit.script
def bin_op_vectorized_invert(a, b, gate_weights):
    """
    Compute 2 binary operations in parallel, then for each operation
    decide (via not_weights) whether to invert the operation's output.

    Args:
        a, b: Tensors of shape (batch, out_dim).
        gate_weights: Tensor of shape (out_dim, 8) representing the mixture
            weights for the 8 operations.
        not_weights: Tensor of shape (out_dim, 2) representing the mixing
            weights for choosing between the base op and its inversion.
            (E.g. via softmax, where index 0 corresponds to not-inverting and
            index 1 to inverting.)

    Returns:
        result: Tensor of shape (batch, out_dim), computed as a weighted sum
            of the effective operations.
    """
    # Precompute common subexpressions
    op0 = a
    op1 = 1 - a
    # Stack operations along a new dimension: shape (batch, out_dim, 8)
    ops = torch.stack([op0, op1], dim=-1)

    # Compute the inverted version of each op: simply 1 - op.
    effective_ops = ops

    # Expand gate_weights (shape (out_dim, 8)) to (1, out_dim, 8)
    gate_weights = gate_weights.unsqueeze(0)

    # Mix the 8 effective operations per neuron.
    result = torch.sum(effective_ops * gate_weights, dim=-1)  # shape (batch, out_dim)
    return result


@torch.jit.script
def bin_op_vectorized_not(a, b, gate_weights, not_weights):
    """
    Compute 8 binary operations in parallel, then for each operation
    decide (via not_weights) whether to invert the operation's output.

    Args:
        a, b: Tensors of shape (batch, out_dim).
        gate_weights: Tensor of shape (out_dim, 8) representing the mixture
            weights for the 8 operations.
        not_weights: Tensor of shape (out_dim, 2) representing the mixing
            weights for choosing between the base op and its inversion.
            (E.g. via softmax, where index 0 corresponds to not-inverting and
            index 1 to inverting.)

    Returns:
        result: Tensor of shape (batch, out_dim), computed as a weighted sum
            of the effective operations.
    """
    # Precompute common subexpressions
    ab = a * b
    ab2 = 2 * ab

    # Define 8 base operations (each will later be paired with its inverse).
    # The chosen operations are one representative from each complementary pair:
    op0 = torch.zeros_like(a)  # 0; inversion yields 1.
    op1 = ab  # A and B; inversion: 1 - (a*b)
    op2 = a - ab  # not(A implies B); inversion: 1 - a + ab
    op3 = a  # A; inversion: 1 - a
    # op4 = b - ab                 # not(B implies A); inversion: 1 - b + ab
    # op5 = b                      # B; inversion: 1 - b
    op6 = a + b - ab2  # A xor B; inversion: 1 - a - b + 2ab
    op7 = a + b - ab  # A or B; inversion: 1 - a - b + ab

    # Stack operations along a new dimension: shape (batch, out_dim, 8)
    ops = torch.stack([op0, op1, op2, op3, op6, op7], dim=-1)

    # Compute the inverted version of each op: simply 1 - op.
    inv_ops = 1 - ops
    # if disable_not:

    # not_weights: shape (out_dim, 2). We expand to (1, out_dim, 2)
    not_weights = not_weights.unsqueeze(0)

    # For each op candidate, choose a mixture between the base op and its inverse.
    # Here not_weights[..., 0] (non-inverted) and not_weights[..., 1] (inverted)
    # are broadcast to the op dimension.
    effective_ops = not_weights[..., 0:1] * ops + not_weights[..., 1:2] * inv_ops
    # effective_ops now has shape (batch, out_dim, 8)

    # Expand gate_weights (shape (out_dim, 8)) to (1, out_dim, 8)
    result = torch.einsum("bod,od->bo", ops, gate_weights)
    # gate_weights = gate_weights.unsqueeze(0)

    # Mix the 8 effective operations per neuron.
    # result = torch.sum(effective_ops * gate_weights, dim=-1)  # shape (batch, out_dim)

    # print(torch.allclose(result, result_t))

    return result


@torch.jit.script
def bin_op_vectorized(a, b, weights):
    """
    Compute all 16 binary operations in parallel and combine them using the given weights.

    a, b: tensors of shape (batch, out_dim)
    weights: tensor of shape (out_dim, 16)
    Returns:
        result: tensor of shape (batch, out_dim)
    """
    # Compute common subexpression only once
    ab = a * b
    ab2 = 2 * ab
    # Define the 16 binary operations using the precomputed ab
    op0 = torch.zeros_like(a)  # 0
    op1 = ab  # A and B
    op2 = a - ab  # not(A implies B)
    op3 = a  # A
    op4 = b - ab  # not(B implies A)
    op5 = b  # B
    op6 = a + b - ab2  # A xor B
    op7 = a + b - ab  # A or B
    op8 = 1 - (a + b - ab)  # not(A or B)
    op9 = 1 - (a + b - ab2)  # not(A xor B)
    op10 = 1 - b  # not(B)
    op11 = 1 - b + ab  # B implies A
    op12 = 1 - a  # not(A)
    op13 = 1 - a + ab  # A implies B
    op14 = 1 - ab  # not(A and B)
    op15 = torch.ones_like(a)  # 1

    # Stack all operations: resulting shape (batch, out_dim, 16)
    ops = torch.stack(
        [
            op0,
            op1,
            op2,
            op3,
            op4,
            op5,
            op6,
            op7,
            op8,
            op9,
            op10,
            op11,
            op12,
            op13,
            op14,
            op15,
        ],
        dim=-1,
    )

    # Expand weights to include a batch dimension: (1, out_dim, 16)
    result = torch.einsum("bod,od->bo", ops, weights)
    # weights = weights.unsqueeze(0)

    # Compute the weighted sum over the operation dimension
    # print(ops.shape)
    # print(weights.shape)
    # result = torch.sum(ops * weights, dim=-1)

    # print(torch.allclose(result, result_t))
    return result


from z3 import *


def gate_expression_16(op_id, A_expr, B_expr):
    """
    Return a Z3 expression corresponding to the operator with ID `op_id`
    applied to (A_expr, B_expr).

    Table references:
      0 -> 0
      1 -> A and B
      2 -> not(A implies B)
      3 -> A
      4 -> not(B implies A)
      5 -> B
      6 -> A xor B
      7 -> A or B
      8 -> not(A or B)
      9 -> not(A xor B)
      10 -> not(B)
      11 -> B implies A
      12 -> not(A)
      13 -> A implies B
      14 -> not(A and B)
      15 -> 1
    """
    if op_id == 0:  # 0
        return z3.BoolVal(False)
    elif op_id == 1:  # A and B
        return And(A_expr, B_expr)
    elif op_id == 2:  # not(A implies B) -> A & not B
        return And(A_expr, Not(B_expr))
    elif op_id == 3:  # A
        return A_expr
    elif op_id == 4:  # not(B implies A) -> B & not A
        return And(B_expr, Not(A_expr))
    elif op_id == 5:  # B
        return B_expr
    elif op_id == 6:  # A xor B
        return Xor(A_expr, B_expr)
    elif op_id == 7:  # A or B
        return Or(A_expr, B_expr)
    elif op_id == 8:  # not(A or B)
        return Not(Or(A_expr, B_expr))
    elif op_id == 9:  # not(A xor B)
        return Not(Xor(A_expr, B_expr))
    elif op_id == 10:  # not(B)
        return Not(B_expr)
    elif op_id == 11:  # B implies A
        return Implies(B_expr, A_expr)
    elif op_id == 12:  # not(A)
        return Not(A_expr)
    elif op_id == 13:  # A implies B
        return Implies(A_expr, B_expr)
    elif op_id == 14:  # not(A and B)
        return Not(And(A_expr, B_expr))
    elif op_id == 15:  # 1
        return z3.BoolVal(True)
    else:
        raise ValueError(f"Invalid operator ID: {op_id}")


def gate_expression_6(op_id, A_expr, B_expr):
    # 0,1,2,3,5,6
    if op_id == 0:  # 0
        return z3.BoolVal(False)
    elif op_id == 1:  # A and B
        return And(A_expr, B_expr)
    elif op_id == 2:  # not(A implies B) -> A & not B
        return And(A_expr, Not(B_expr))
    elif op_id == 3:  # A
        return A_expr
    elif op_id == 4:  # not(B implies A) -> B & not A
        return Xor(A_expr, B_expr)
    elif op_id == 5:  # B
        return Or(A_expr, B_expr)
    else:
        raise ValueError(f"Invalid operator ID: {op_id}")


def gate_expression_not(op_id, A_expr, B_expr):
    if op_id == 0:  # 0
        return A_expr
    elif op_id == 1:  #
        return Not(A_expr)
    else:
        raise ValueError(f"Invalid operator ID: {op_id}")


class BaseLogicLayer(nn.Module):
    """
    A hardened version of a scalable differentiable logic gate network layer with fixed connections and gates.

    In this version, the forward pass is hardened: a single connection per output is selected
    (using a fixed index) rather than a soft probability distribution. The gate weights are learned as before.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str = "cpu",
        grad_factor: float = 1.0,
        initalization: str = "random",
        seed=None,
        gate_set: Literal[2, 6, 16] = 16,     # choices between 6 and 16 (2 is inverter Layer)
        **kwargs,
    ):
        """
        :param in_dim:         Input dimensionality of the layer.
        :param out_dim:        Output dimensionality of the layer.
        :param num_connections:Number of random input connections per output.
        :param device:         Device (e.g., 'cuda' or 'cpu').
        :param grad_factor:    Gradient factor.
        :param hardened_index: Fixed connection index (0 <= index < num_connections) to use for every output.
        :param initalization:  Initialization method for weights ('random' or 'residual').
        :param seed:           Optional random seed.
        :param extra_not:      If True, additional NOT weights are created.
        :param disable_not:    If True, the extra NOT branch is disabled.
        :param freeze_interconnect: Unused in hardened mode.
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        if gate_set == 6:
            self.bin_op = bin_op_vectorized_base
        elif gate_set == 16:
            self.bin_op = bin_op_vectorized
        elif gate_set == 2:
            self.bin_op = bin_op_vectorized_invert
        self.gate_set = gate_set

        selected = torch.zeros((self.out_dim, 2), dtype=torch.long)

        c = torch.randperm(2 * self.out_dim) % self.in_dim
        c = torch.randperm(self.in_dim)[c]
        c = c.reshape(self.out_dim, 2)
        selected[:, :2] = c
        self.register_buffer("selected_inputs", selected)
        self.selected_inputs = self.selected_inputs.to(device)
        self.weights_size = gate_set
        if initalization == "random":
            self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device)*0.1)
            #init_tensor = torch.rand(out_dim, self.weights_size, device=device)
            #for i in range(out_dim):
            #    rand_idx = torch.randint(0, self.weights_size, (1,)).item()
            #    init_tensor[i, rand_idx] *= 5.0
            #self.weights = nn.Parameter(init_tensor)

        elif initalization == "residual":
            weight_on_a = 2.0
            weight_on_rest = 1.0
            # print(out_dim)
            # print(self.weights_size)
            weight_vec = (
                torch.ones(out_dim, self.weights_size, device=device) * weight_on_rest
            )
            weight_vec[:, 3] = weight_on_a
            weight_vec = weight_vec / weight_vec.sum(dim=-1, keepdim=True)
            self.weights = nn.Parameter(weight_vec)
        else:
            raise ValueError("Only 'random' and 'residual' initalization is supported.")

        self.num_neurons = out_dim
        self.num_weights = out_dim

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, gate_set={self.weights_size}"

    def forward(self, x):
        """
        Hardened forward pass: uses fixed (hardened) connection indices.
        Instead of computing a soft distribution over candidate connections, this function
        directly selects one connection per output using the fixed hardened_index.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_dim].

        Returns:
            torch.Tensor: Output tensor after applying the binary logic operation.
        """
        # x: [batch_size, in_dim]
        batch_size = x.size(0)
        # print(batch_size)
        # For each output row, use the fixed hardened_index to select a single input connection.
        indices_a = self.selected_inputs[:, 0]
        indices_b = self.selected_inputs[:, 1]
        a = x[:, indices_a]
        b = x[:, indices_b]
        assert a.shape == b.shape == (batch_size, self.weights.shape[0]), (
            a.shape,
            b.shape,
            self.weights.shape,
        )
        if self.training:
            # During training, use the weights to compute a weighted sum of the operations.
            gate_weights = F.softmax(self.weights, dim=-1)
        else:
            gate_weights = F.one_hot(
                self.weights.argmax(dim=-1), num_classes=self.weights_size
            ).float()
        output = self.bin_op(a, b, gate_weights)
        return output

    def forward_z3_logic(self, input_z3_expressions):
        """
        Given a list (or array) of Z3 expressions for the inputs (of length self.in_dim),
        compute the output Z3 expression for each gate in this layer.
        For each gate, use the fixed connection indices from selected_inputs:
            A_expr = input_z3_expressions[selected_inputs[i, 0]]
            B_expr = input_z3_expressions[selected_inputs[i, 1]]
        Then, using the gate's operator (given by the argmax of weights[i]) and the appropriate gate_expression
        function (based on self.gate_set), compute the gate's Z3 expression.
        Returns a list of Z3 expressions (of length self.out_dim).
        """
        # Extract indices (as numpy arrays)
        a_indices = self.selected_inputs[:, 0].cpu().numpy()
        b_indices = self.selected_inputs[:, 1].cpu().numpy()
        op_ids = self.weights.argmax(dim=-1).cpu().numpy().astype(int)

        # Choose the proper gate_expression function.
        if self.gate_set == 2:
            gate_expression_func = gate_expression_not
        elif self.gate_set == 16:
            gate_expression_func = gate_expression_16
        elif self.gate_set == 6:
            gate_expression_func = gate_expression_6
        else:
            raise ValueError(f"Invalid gate_set: {self.gate_set}")

        output_exprs = []
        # print(len(input_z3_expressions))
        for i in range(len(a_indices)):
            # print(a_indices[i])
            # print(b_indices[i])
            A_expr = input_z3_expressions[a_indices[i]]
            B_expr = input_z3_expressions[b_indices[i]]
            op_id = op_ids[i]
            out_expr = gate_expression_func(op_id, A_expr, B_expr)
            output_exprs.append(out_expr)
        return output_exprs
    
class BaseGroupSum(nn.Module):
    """
    Hardened version of GroupSumLearnable.

    In this version the interconnect is fixed:
      - The fixed connection indices are stored in hardened_indices (one index per output).
      - A fake logits_b is created by simply storing logits_a (unused).

    The forward pass uses the fixed indices (converted into one-hot vectors)
    to gather from the input and compute the output.
    """

    def __init__(
        self,
        k: int,
        tau: float = 1.0,
        beta: float = 0.0,
        in_dim: int = None,
        device: str = "cpu",
        highest_multiplicity: int = 1,
        grad_factor: float = 1.0,
    ):
        super().__init__()
        assert in_dim is not None and in_dim is not None
        self.k = k
        self.tau = tau
        self.beta = beta
        self.device = device
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.highest_multiplicity = highest_multiplicity
        self.grad_factor = grad_factor
        self.num_connections = 1

        # Initialize weight as in the learnable version.
        if highest_multiplicity == 1:
            weight = torch.ones(self.out_dim, dtype=torch.float32, device=device)
        else:
            base_count = self.out_dim // highest_multiplicity
            remainder = self.out_dim % highest_multiplicity
            counts = [base_count] * highest_multiplicity
            for i in range(remainder):
                counts[i] += 1
            weights_list = []
            for i, count in enumerate(counts):
                weight_val = highest_multiplicity - i
                weights_list.extend([weight_val] * count)
            weight = torch.tensor(weights_list, dtype=torch.float32, device=device)
        self.register_buffer("weight", weight)
        # Create the fixed selected_inputs buffer (these were used to pick candidate connections).
        self.register_buffer(
            "selected_inputs",
            torch.zeros((self.out_dim, self.num_connections), dtype=torch.long),
        )
        c = torch.arange(in_dim, device=device).reshape(in_dim, 1)
        self.selected_inputs[:, 0] = c[:, 0]
        self.return_grouped = False

    def forward(self, x):
        batch_size = x.size(0)
        selected_inputs_expanded = self.selected_inputs.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        selected_a = torch.gather(
            x.unsqueeze(-1).expand(
                -1, -1, self.num_connections
            ),
            dim=1,
            index=selected_inputs_expanded[..., :1],
        )
        x = selected_a.squeeze(-1)
        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        x = x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k)
        x_weighted = x  # * self.weight
        if self.return_grouped:
            return x_weighted

        return x_weighted.sum(-1) / self.tau + self.beta

    def extra_repr(self):
        return f"k={self.k}, tau={self.tau}, highest_multiplicity={self.highest_multiplicity}, in_dim={self.in_dim}"
