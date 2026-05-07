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
    various utility functions.
"""

from .difflogic import BaseLogicLayer, BaseGroupSum
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import torch

def get_model_infos(model):
    inital_layer_size = -1
    last_layer_out_dim = 0
    for name, module in model.named_modules():
        if isinstance(module, BaseLogicLayer):
            if inital_layer_size == -1:
                inital_layer_size = module.in_dim
            
        if isinstance(module, BaseGroupSum):
            classes= module.k
            last_layer_out_dim = module.in_dim
    return inital_layer_size, last_layer_out_dim, classes

def compute_probabilities(tensor):
    """
    Compute the probability of each index 0, 1, 2 by dividing it by the sum over axis 1.
    """
    
    sum_over_axis = torch.sum(tensor, 1, keepdims=True)
    probabilities = tensor / sum_over_axis
    return probabilities

def decompose_print_diff(x, xp, dict_numeric, dict_categorical):
    """
    x, xp : lists (or arrays) of bits (0/1), length at least the max 'start+length'
    dict_numeric, dict_categorical : lists of dicts, each with keys:
       - 'col': int (column index)
       - 'start': int
       - 'length': int
       - optionally 'thermometer': bool for numeric

    We extract the bits from x and xp for each column's block,
    build a string like '0010', and if they differ, print info.
    """
    # Combine numeric + categorical info so we can iterate over all columns
    # in a single loop. Alternatively, you could do them separately.
    all_info = dict_numeric + dict_categorical

    # Sort by 'col' if you prefer to see them in ascending column order.
    # If you want the order they appear in your dictionary, skip the sort.
    # all_info.sort(key=lambda d: d['col'])

    for item in all_info:
        col = item['col']
        start = item['start']
        length = item['length']

        # Extract the slice for x
        x_block = x[start : start + length]
        # Convert to a string like '0010'
        x_str = ''.join(str(int(b)) for b in x_block)

        # Extract the slice for xp
        xp_block = xp[start : start + length]
        xp_str = ''.join(str(int(b)) for b in xp_block)

        if x_str != xp_str:
            # Print only if they differ
            # You can print or store in a dict as you wish
            # if col is a name 
            
            print(f"Attribute {col} | start={start:3d}, length={length:2d} | "
                  f"x_result='{x_str}'  xp_result='{xp_str}'")

def parse_counterexample(m, numeric_dict,cat_dict, with_xp=True, C=2, in_dim=400,out_dim=102):
    #print("Raw Counterexample:", m)
    N = in_dim
    # Dictionaries to hold index -> value mappings
    x_vals = np.full(N, 0, dtype=np.int32)
    xp_vals = np.full(N, 0, dtype=np.int32) if with_xp else None

    x_out_vals = np.full(out_dim, 0, dtype=np.int32)
    xp_out_vals = np.full(out_dim, 0, dtype=np.int32)

    # Regular expression to parse variable names with multi-word prefixes
    regex = re.compile(r'^(x|xp|x_out|xp_out)_(\d+)$')

    # 1. Collect all model entries in the appropriate dictionary
    for d in m.decls():
        var_name = d.name()  # e.g., "x_5", "xp_43", "x_out_88", etc.
        val = m[d]           # True/False or some other Z3 value

        match = regex.match(var_name)
        if match:
            prefix = match.group(1)      # Extract the prefix (e.g., "x", "x_out")
            index = int(match.group(2))  # Extract the index as an integer

            if prefix == 'x':
                x_vals[index] = int(bool(val))
            elif prefix == 'xp':
                xp_vals[index] = int(bool(val))
            elif prefix == 'x_out':
                x_out_vals[index] = int(bool(val))
            elif prefix == 'xp_out':
                xp_out_vals[index] = int(bool(val))
        else:
            # Handle unexpected variable names (optional)
            # this is not really used as (correct!) sub formulas may have other names
            # print(f"Unexpected variable name format: {var_name}")
            pass

    x_array = x_vals
    x_out_array = x_out_vals

    if with_xp:
        xp_array = xp_vals
        xp_out_array = xp_out_vals

    # 3. Print out the arrays
    #print("x =", np.array(x_array))
    #print("xp =", np.array(xp_array))
    # print where there are differences in x and xp
    if with_xp:
        diffs =  np.array(x_array) != np.array(xp_array)
    # display as int for easier visibility
    
        decompose_print_diff(x_array, xp_array, numeric_dict, cat_dict)
    
        print("differences x/xp =",diffs.astype(np.int32))
        print(np.where(diffs)[0])
    #print("x_out =", np.array(x_out_array))
    #print("xp_out =",  np.array(xp_out_array))
    # print voting power for class 1 in x_out
    
    
    #print("Voting power for class 1 in x_out: ", np.sum(np.array(x_out_array[:len(x_out_array)//2])))
    #print("Voting power for class 2 in x_out: ", np.sum(np.array(x_out_array[len(x_out_array)//2:])))
    # split into C groups
    for i in range(C):
        print(f"Voting power for class {i} in x_out: ", np.sum(np.array(x_out_array[i*len(x_out_array)//C:(i+1)*len(x_out_array)//C])))
        
    if with_xp:
        for i in range(C):
            print(f"Voting power for class {i} in xp_out: ", np.sum(np.array(xp_out_array[i*len(x_out_array)//C:(i+1)*len(x_out_array)//C])))
        
        #print("Voting power for class 1 in xp_out: ", np.sum(np.array(xp_out_array[:len(xp_out_array)//2])))
        #print("Voting power for class 2 in xp_out: ", np.sum(np.array(xp_out_array[len(xp_out_array)//2:])))
        print("Example xp: ", np.array(xp_array).astype(np.int32))
    else:
        xp_array = None
        
    print("Example x: ", np.array(x_array).astype(np.int32))
    
    return np.array(x_array), np.array(xp_array)

def eval_model(model, loader, mode, device='cpu'):
    orig_mode = model.training
    res = 0
    model.eval()
    unique_predictions = []
    with torch.no_grad():
        model.train(mode=mode)
        #                (model(x.to(device).round()).argmax(-1) == y.to(device)).to(torch.float32).mean().item()
        #        for x, y in loader
        for x,y in tqdm(loader):
            # print(x)
            # print(y)
            x = x.to(device).round().float()
            #print(x[0,0,10:20,10:20])
            #print(x.shape)
            #exit()
            y = y.to(device)
            outs = model(x).argmax(-1)
            unique_predictions.append(np.unique(outs.cpu().numpy()))
            targets = y
            res += (outs == targets).to(torch.float32).mean()
        res = res / len(loader)
        model.train(mode=orig_mode)
        # sum up the unique predictions
        all_unique_predictions = np.unique(np.concatenate(unique_predictions))
        print(f"Unique predictions: {all_unique_predictions} (number of unique predictions: {len(all_unique_predictions)})")
        #for i in range(len(unique_predictions[0])):
        #    print(f"Unique prediction {unique_predictions[0][i]}: {unique_predictions[1][i]}")
    return res.item()


def run_ce_trough_model(model, x):
    x = torch.tensor(x).float().unsqueeze(0)
    # compute the length of the model
    layer_number = len(model)
    model[layer_number-1].return_grouped = True
    out = model(x)
    model[layer_number-1].return_grouped = False
    print("model results", torch.sum(out,axis=2))
    return torch.sum(out,axis=2).detach().numpy()
