#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logic Gate Neural Networks are Good for Verification
Accepted at NeuS ’25
Paper: Kresse, F., Yu, E., Lampert, C. H., & Henzinger, T. A. (2025).

Code authors:
    Fabian Kresse (corresponding) – fabian.kresse@ist.ac.at
    Emily Yu – emily.yu@ist.ac.at

Description:
    Main script to run the verification experiments.
"""

import z3
from z3 import Bool, Not, Or, And, Implies, Xor
from .sorting_network import sort_net
from .difflogic import gate_expression_not, gate_expression_16, gate_expression_6
from .difflogic import BaseLogicLayer, BaseGroupSum
# import pbeq from z3
from z3 import PbEq

def encode_input_assignment(x_assignment, z3_input_expressions_x):
    """
    Given a tensor of input assignments (with 0.0/1.0 values) and a dictionary
    mapping indices to Z3 Bool variables, return a dictionary with a single
    constraint that forces each variable to take the corresponding value.

    :param x_assignment: a PyTorch tensor (or similar) of 0.0/1.0 values.
    :param z3_input_expressions_x: dictionary mapping indices to Z3 Bool variables.
    :return: dictionary with key "constraints" containing a list with one big constraint.
    """
    constraints = []
    # Loop over each variable in the dictionary.
    for idx, var in z3_input_expressions_x.items():
        val = x_assignment[idx].item()  # Get value as Python float.
        if val == 1.0:
            constraints.append(var)
        else:
            constraints.append(Not(var))

    # Combine all individual constraints into one big conjunction.
    big_constraint = And(constraints)
    return {"constraints": [big_constraint]}


def init_z3_expressions_for_inputs(num_inputs, network="x"):
    """
    Create num_inputs Boolean variables in Z3,
    and return them in a dictionary (or list) indexed 0..num_inputs-1.
    Offset for two or more networks
    """
    z3_vars = {}
    for i in range(num_inputs):
        var_name = f"{network}_{i}"
        z3_vars[i] = Bool(var_name)
    return z3_vars


def encode_sat_model_with_vars_base_layers(
    model, z3_expressions_input, name="x", output_name="out"
):
    """
    Similar to encode_sat_model, but for each gate, we create a
    fresh boolean variable gate_var instead of returning the entire formula.
    Then we add a constraint gate_var == gate_expression(op_id, A_expr, B_expr).

    :param model: list of layers
    :param layers: list of layer indices, e.g. [1,2,3]
    :param z3_expressions_input: a dict or list of z3 Bools for the "previous" wires
    :param constraints: a list of constraints we can append to
    :return: (dictionary_of_gate_vars_per_layer, final_mapping_for_next_layer)
    """
    # We'll store gate_vars_per_layer[layer_idx] = list of fresh Bool vars
    gate_vars_per_layer = {}
    constraint = {"constraints": []}
    # This holds the variables for the previous layer's outputs
    z3_expressions_prev_layer = z3_expressions_input

    # loop over the model
    layer_idx = 0
    for layer in model:
        if isinstance(layer, BaseLogicLayer):
            a_indices = layer.selected_inputs[:, 0].cpu().numpy()  # shape: (num_gates,)
            b_indices = layer.selected_inputs[:, 1].cpu().numpy()
            op_ids = layer.weights.argmax(dim=-1).cpu().numpy()
            gate_set = layer.gate_set

            if gate_set == 2:
                gate_expression_func = gate_expression_not
            elif gate_set == 16:
                gate_expression_func = gate_expression_16
            elif gate_set == 6:
                gate_expression_func = gate_expression_6
            else:
                raise ValueError(f"Invalid gate_set: {gate_set}")
            z3_expressions_curr = []
            for g_idx in range(len(a_indices)):
                A_expr = z3_expressions_prev_layer[a_indices[g_idx]]
                B_expr = z3_expressions_prev_layer[b_indices[g_idx]]
                op_id = int(op_ids[g_idx])
                z3_expr = gate_expression_func(op_id, A_expr, B_expr)
                # create a new variable that is forced equal to the z3_expr
                new_gate = z3.Bool(f"{name}_{layer_idx}_gate_{g_idx}")
                z3_expressions_curr.append(new_gate)
                constraint["constraints"].append(new_gate == z3_expr)
            z3_expressions_prev_layer = z3_expressions_curr
            layer_idx += 1
        if isinstance(layer, BaseGroupSum):
            a_indices = layer.selected_inputs[:, 0].cpu().numpy()
            z3_expressions_out = []
            for g_idx in range(len(a_indices)):
                A_expr = z3_expressions_prev_layer[a_indices[g_idx]]
                new_gate = z3.Bool(f"{name}_{output_name}_{g_idx}")
                z3_expressions_out.append(new_gate)
                constraint["constraints"].append(new_gate == A_expr)
    return constraint, z3_expressions_out


def compute_relation_in_out_constraint(
    final_layer_expressions_x,
    final_layer_expressions_xp,
    z3_input_expressions_x,
    z3_input_expressions_xp,
    final_output_deps,
):
    """
    Build constraints such that:
    For each final output 'o', if all its dependency inputs match between x and xp,
    then the output bit also must match: x_out_o == xp_out_o.
    This is not used in the paper, but antectodally speeds up the solver by roughly 5%.
    Expect a higher speedup for larger networks, and especially deeper networks.

    :param final_layer_expressions_x: dict of {gate_idx: z3 Bool} for final outputs (x-network).
    :param final_layer_expressions_xp: dict of {gate_idx: z3 Bool} for final outputs (xp-network).
    :param z3_input_expressions_x: dict of {input_idx: z3 Bool} for input bits x_0..x_N.
    :param z3_input_expressions_xp: dict of {input_idx: z3 Bool} for input bits xp_0..xp_N.
    :param final_output_deps: dict of {gate_idx: set_of_input_indices} telling which inputs feed gate_idx.
    :return: dict with key 'constraints', containing a list of Z3 implications.
    """
    constraints = []

    for out_idx, deps in final_output_deps.items():
        # Gather the input equality conditions
        input_equalities = []
        for d in deps:
            input_equalities.append(
                z3_input_expressions_x[d] == z3_input_expressions_xp[d]
            )

        if len(input_equalities) == 0:
            # If the gate claims no dependencies, skip or just unify outputs (optional).
            # For safety, we might do out_x == out_xp if truly no input dependency.
            conj_of_equalities = True  # Then the implies is unconditional
        else:
            conj_of_equalities = z3.And(*input_equalities)

        # Output equality
        same_output = (
            final_layer_expressions_x[out_idx] == final_layer_expressions_xp[out_idx]
        )

        # Add implication: if all inputs match, outputs match
        constraints.append(z3.Implies(conj_of_equalities, same_output))

    return {"constraints": constraints}


def encode_class_diff_general_pure_sat(o_x, o_xp, N, C):
    """
    Enforce f(x) != f(xp) for general number of classes (C > 2),
    with tie-breaking such that lower-index class wins in case of a tie.

    :param o_x: List of z3.Bool variables representing outputs for x
    :param o_xp: List of z3.Bool variables representing outputs for xp
    :param N: Number of outputs per class
    :param C: Number of classes
    :return: Dictionary containing Z3 constraints ensuring f(x) != f(xp)
    """
    # Build sum of outputs for each class in x and xp
    sorted_x = []
    sorted_xp = []
    consistency_constraints = []
    for i in range(C):
        # Extract blocks of size N for each class
        block_x = o_x[i * N : (i + 1) * N]
        block_xp = o_xp[i * N : (i + 1) * N]
        sort_net(block_x)
        sort_net(block_xp)

        # Compute the sum of bits in each class

        sorted_x.append(block_x)
        sorted_xp.append(block_xp)

    # Encode class-winning with tie-breaking (lower index wins ties)
    class_diff_constraints = []
    for i in range(C):
        class_i_wins_x = []
        # i has greater or equal sum compared to all lower-index classes
        for j in range(i):
            for big, small in zip(sorted_x[i], sorted_x[j]):
                class_i_wins_x.append(z3.Implies(small, big))
        # i has greater or equal sum compared to all higher-index classes
        for j in range(i + 1, C):
            at_least_one_greater = []  # for strict inequality
            for big, small in zip(sorted_x[i], sorted_x[j]):
                at_least_one_greater.append(z3.And(big, z3.Not(small)))
            class_i_wins_x.append(z3.Or(at_least_one_greater))
        class_i_wins_x = z3.And(class_i_wins_x)

        other_class_wins_xp = []
        for j in range(C):
            if j == i:
                continue
            class_j_wins_xp = []
            # j has greater or equal sum compared to all lower-index classes
            for k in range(j):
                for big, small in zip(sorted_xp[j], sorted_xp[k]):
                    class_j_wins_xp.append(z3.Implies(small, big))

            # j has strictly greater sum compared to all higher-index classes
            for k in range(j + 1, C):
                at_least_one_greater = []  # for strict inequality
                for big, small in zip(sorted_xp[j], sorted_xp[k]):
                    at_least_one_greater.append(z3.And(big, z3.Not(small)))
                class_j_wins_xp.append(z3.Or(at_least_one_greater))
            other_class_wins_xp.append(z3.And(class_j_wins_xp))
        other_class_wins_xp = z3.Or(other_class_wins_xp)

        # Add the constraint: if i wins for x, another class wins for xp
        class_diff_constraints.append(z3.Implies(class_i_wins_x, other_class_wins_xp))

    # Return the constraints in a dictionary format
    constraints = {"constraints": class_diff_constraints}
    return constraints


def build_input_domain_constraints(z3_input_vars, categorical_info, numeric_info):
    """
    Build domain constraints for all possible valid inputs, returning them
    in a dictionary without adding them to a solver.

    :param z3_input_vars: A dict or list of z3 Bool variables,
                          indexed 0..(N-1), e.g. from init_z3_expressions_for_inputs.
    :param categorical_info: a list of dicts describing each categorical column's encoding:
        [
           {
             "col": <column index or name>,
             "start": <int index in z3_input_vars>,
             "length": <int number of bits>
           },
           ...
        ]
        We'll enforce exactly one bit is True in each block.

    :param numeric_info: a list of dicts describing each numeric column's thermometer encoding:
        [
           {
             "col": <column index or name>,
             "start": <int index in z3_input_vars>,
             "length": <int number of bits>,
             "thermometer": <bool> (True means we enforce prefix-of-1 constraint)
           },
           ...
        ]
        We'll enforce x[i] >= x[i+1] in boolean sense for each adjacent pair if 'thermometer' is True.

    :return: A dictionary like:
        {
          "constraints": [z3_expr_1, z3_expr_2, ...]
        }
        where each z3_expr_i is a constraint that you can later add to a solver.
    """
    constraints = []

    # 1) CATEGORICAL columns: exactly one bit True in the block
    for cat in categorical_info:

        start = cat["start"]
        length = cat["length"]
        block_vars = []
        for i in range(length):
            block_vars.append(z3_input_vars[start + i])

        # (a) "At least one" is True
        constraints.append(Or(*block_vars))

        # (b) "No two can be True at the same time"
        for i in range(length):
            for j in range(i + 1, length):
                constraints.append(Or(Not(block_vars[i]), Not(block_vars[j])))

    # 2) NUMERIC columns: "thermometer" => prefix of 1's
    for num in numeric_info:
        start = int(num["start"])
        length = int(num["length"])  # TODO! fix this where it happens
        if num.get("thermometer", True):
            # For each pair (x[i], x[i+1]), enforce x[i] >= x[i+1]
            # => not(x_i=1, x_{i+1}=0)
            # => Or(Not(x_i), x_{i+1})
            for i in range(length - 1):
                xi = z3_input_vars[start + i]
                xip1 = z3_input_vars[start + i + 1]
                constraints.append(Implies(xip1, xi))
        else:
            raise ValueError("How did this happen?")
    return {"constraints": constraints}


from z3 import Or, Not


def encode_temperature_constraints_pure_sat(
    z3_input_vars_x, z3_input_vars_xp, numerical_info, epsilon
):
    """
    Build constraints enforcing that for each numeric (thermometer) block in `categorical_info`,
    the integer sum of bits for x and x' differ by at most epsilon.

    :param z3_input_vars_x: list/dict of z3 Bool variables for input x
    :param z3_input_vars_xp: list/dict of z3 Bool variables for input x'
    :param categorical_info: a list of dicts describing numeric columns, each like:
        {
          "start": <int>,
          "length": <int>,
          ... possibly other fields ...
        }
        (Although named 'categorical_info' here, this is presumably the structure for
         numeric/thermometer blocks.)
    :param epsilon: an integer specifying how many bits difference is allowed
                   between x and x' for each numeric block.
    :return: A dictionary {"constraints": [...z3 expressions...]}
             so you can merge them into your overall SAT/SMT encoding.

    For each block:
      let i_x = sum of bits in x
      let i_xp = sum of bits in x'
      enforce |i_x - i_xp| <= epsilon.
    """
    constraints = []

    for block in numerical_info:
        start = block["start"]
        length = block["length"]

        # Build the sum of bits for x in this block
        bits_x = [z3_input_vars_x[start + i] for i in range(length)]
        # sorting here should not be necessary, as the inputs are already 'sorted' by design and this is checked
        # leaving it in as we had it like this in the submission...
        sort_net(bits_x)

        # Build the sum of bits for x' in this block
        bits_xp = [z3_input_vars_xp[start + i] for i in range(length)]
        # same here
        sort_net(bits_xp)

        assert len(bits_x) == len(bits_xp)
        # Enforce -epsilon <= sum_x - sum_xp <= epsilon
        # which is sum_x - sum_xp <= epsilon  AND  sum_xp - sum_x <= epsilon
        # constraints.append(sum_x - sum_xp <= epsilon)
        for i in range(epsilon, len(bits_x)):
            constraints.append(z3.Implies(bits_x[i], bits_xp[i - epsilon]))
        # constraints.append(sum_xp - sum_x <= epsilon)
        for i in range(epsilon, len(bits_xp)):
            constraints.append(z3.Implies(bits_xp[i], bits_x[i - epsilon]))

    return {"constraints": constraints}


def encode_temperature_constraints(
    z3_input_vars_x, z3_input_vars_xp, numerical_info, epsilon
):
    """
    Build constraints enforcing that for each numeric (thermometer) block in `categorical_info`,
    the integer sum of bits for x and x' differ by at most epsilon.

    :param z3_input_vars_x: list/dict of z3 Bool variables for input x
    :param z3_input_vars_xp: list/dict of z3 Bool variables for input x'
    :param categorical_info: a list of dicts describing numeric columns, each like:
        {
          "start": <int>,
          "length": <int>,
          ... possibly other fields ...
        }
        (Although named 'categorical_info' here, this is presumably the structure for
         numeric/thermometer blocks.)
    :param epsilon: an integer specifying how many bits difference is allowed
                   between x and x' for each numeric block.
    :return: A dictionary {"constraints": [...z3 expressions...]}
             so you can merge them into your overall SAT/SMT encoding.

    For each block:
      let i_x = sum of bits in x
      let i_xp = sum of bits in x'
      enforce |i_x - i_xp| <= epsilon.
    """
    constraints = []
    # print(numerical_info)
    for block in numerical_info:

        start = int(block["start"])
        length = int(block["length"])

        # Build the sum of bits for x in this block
        bits_x = [z3_input_vars_x[start + i] for i in range(length)]
        sum_x = z3.Sum([z3.If(b, z3.IntVal(1), z3.IntVal(0)) for b in bits_x])

        # Build the sum of bits for x' in this block
        bits_xp = [z3_input_vars_xp[start + i] for i in range(length)]
        sum_xp = z3.Sum([z3.If(b, z3.IntVal(1), z3.IntVal(0)) for b in bits_xp])

        # Enforce -epsilon <= sum_x - sum_xp <= epsilon
        # which is sum_x - sum_xp <= epsilon  AND  sum_xp - sum_x <= epsilon
        constraints.append(sum_x - sum_xp <= epsilon)
        constraints.append(sum_xp - sum_x <= epsilon)

    return {"constraints": constraints}


def encode_sensitive_attribute_flip(
    z3_input_vars_x, z3_input_vars_xp, categorical_info, attribute_to_flip=None
):
    """
    This just enforces that this attribute changes
    """
    sat_encoding = {}
    # print(categorical_info)
    constraints = sat_encoding.setdefault("constraints", [])
    found_block = False
    print("Attributes available to flip:")
    print([block["col"] for block in categorical_info])
    gender_block = None
    for block in categorical_info:
        if attribute_to_flip == block["col"]:
            gender_block = block
            found_block = True
            break
    if not found_block and ("None" not in attribute_to_flip):
        raise ValueError("Couldnt find sensitive attribute")

    if "None" in attribute_to_flip:
        print("No sensitive attribute, checking robustness")
    else:
        # print(gender_block)
        start_g = gender_block["start"]
        length_g = gender_block["length"]
        # for the gender block they are not allowed to be the same
        bit_changes = [
            (z3_input_vars_x[start_g + i] != z3_input_vars_xp[start_g + i], 1)
            for i in range(length_g)
        ]

        # Enforce that exactly 2 bits are different using PbEq.
        constraints.append(PbEq(bit_changes, 2))

    for cat in categorical_info:
        if gender_block is not None:
            if cat is gender_block:
                continue
        s = cat["start"]
        ln = cat["length"]
        for i in range(ln):
            # pass
            constraints.append(z3_input_vars_x[s + i] == z3_input_vars_xp[s + i])

    return sat_encoding


def encode_confidence_largest_class_pure_sat(output_vars_x, N, C, kappa):
    """
    Builds constraints for the condition 'confidence(largest_class) > kappa'
    in a C-class model where each class i has N Boolean outputs.

    Confidence is defined as:
        maxClassScore / sumAllScores > kappa
    where:
        - maxClassScore = maximum sum over all class blocks
        - sumAllScores = sum of all class scores

    :param output_vars_x: List of Z3 Bools representing outputs of the final layer
    :param N: Number of outputs per class
    :param C: Number of classes
    :param kappa: Confidence threshold (e.g., 0.75)
    :return: A dictionary containing Z3 constraints and the main confidence condition.
    """
    assert (
        len(output_vars_x) == N * C
    ), f"output_vars_x must have length={N * C}, got {len(output_vars_x)}"

    # We'll create sums for each class
    sorted_x = []
    sorted_x_all = []
    for i in range(C):
        block_x = output_vars_x[i * N : (i + 1) * N]  # Boolean outputs for class i
        sorted_x_all.extend(block_x)
        sort_net(block_x)
        sorted_x.append(block_x)
    sort_net(sorted_x_all)

    constraints = []
    for i in range(len(sorted_x_all)):
        threshold = int((i + 1) * kappa)
        constraints.append(
            z3.Implies(
                sorted_x_all[i],
                z3.Or([c[threshold] for c in sorted_x if len(c) > threshold]),
            )
        )

    # Collect all constraints
    sat_encoding = {"constraints": []}
    sat_encoding["constraints"].extend([z3.And(constraints)])

    return sat_encoding
