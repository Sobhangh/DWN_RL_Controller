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
    The SAT encoding for the logic gate network.
"""
import utils.verification_utils as vutils
import z3
from z3 import And
import time
from z3 import *
import os
import tempfile
import uuid
import subprocess
#from pysat.formula import *
#from pysat.solvers import Cadical195,Solver
from .utils import get_model_infos

def sat_check_model_existence(model, confidence,
                                input_constraints_fn):
    """
    @brief Checks if the confidence threshold can be obtained for the model with some x.
    """
    
    # get the input dimension by looping through
    inital_layer_size, N, C = get_model_infos(model)
    z3_input_expressions_x = vutils.init_z3_expressions_for_inputs(inital_layer_size, "x")
    
    input_constraints = input_constraints_fn(z3_input_expressions_x)
    
    
    model_encoding_x, final_layer_expressions_x = vutils.encode_sat_model_with_vars_base_layers(model,
                                                                                                    z3_input_expressions_x,
                                                                                                    name="x")

    constraint_confidence = vutils.encode_confidence_largest_class_pure_sat(final_layer_expressions_x,
                                                                   N//C,
                                                                   C, 
                                                                   kappa = confidence)
    z3.set_param("parallel.enable", True)

    s = z3.Solver()
    # add more threads
    set_option("parallel.enable", True)
    set_option("parallel.threads.max", 32)
    s.add(model_encoding_x["constraints"])
    # s.add(constraint_output_x["constraints"])
    s.add(input_constraints["constraints"])

    s.add(constraint_confidence["constraints"])
    start = time.time()
    result = s.check()
    end_time = time.time() - start
    print("Time (SAT GR):",end_time )
    if result == z3.unsat:
        return False, None, end_time
    else:
        m = s.model()
        return True, m, end_time

def compute_verified_accuracy(model, results_array, dataset, subset=True):
    model.eval()
    verified_acc = 0
    for batch in tqdm(dataset, desc="Processing batches"):
        X, y = batch
        X = X.round()
        predictions = model(X)
        verified_acc += torch.sum((predictions.argmax(dim=1) == y) & results_array)/len(results_array)
        if subset: 
            break
    if not subset:
        verified_acc = verified_acc / len(dataset)
    else:
        verified_acc = verified_acc
    return verified_acc

from tqdm import tqdm
import torch

def local_robustness(model,
                    x_train, y_train,
                    input_constraints_fn,
                    relational_constraints,
                    parse_ce_gc,
                    run_ce_trough_model,
                    epsilon=1):
    local_robustness_solver, input_expression = sat_check_global_robustness(model,
                                confidence=None,
                                input_constraints_fn= input_constraints_fn,
                                relational_constraints=relational_constraints,
                                return_local_robustness_encoding=True,)
    
    #print("Solver local robustness")
    #print(input_expression)
    model.to("cpu")
    results_array = []   # This could be a list of tuples: (result, model or None)

    # Process each batch in the training set.
    overall_sat_processing_time = 0
    new_samples = []
    n_samples = x_train.shape[0]
    batch_size = 64
    #print("n_samples:", n_samples)
    #print(batch_size)
    #print(x_train.shape)
    for i in tqdm(range(0, n_samples//batch_size), total=n_samples//batch_size):
    #for batch in x_train, y_train:#tqdm(dataset, desc="Processing batches"):
        #print("Batch:", i)
        X, y = x_train[i*batch_size: (i+1) * batch_size],y_train[i*batch_size: (i+1) * batch_size]  # X shape assumed: [batch_size, channels, height, width]
        batch_size = X.size(0)
        #print(X.shape)
        #print(y)
        #print(X)
        
        # Precompute input encodings for the whole batch.
        # We assume each input is in channel 0, and that the image is 28x28 (MNIST).
        # You may need to adjust indices if your data has a different shape.
        batch_encodings = []
        for j in range(batch_size):
            #print(X.shape)
            #print(X)
            # print unique values in X
            #print("X unique values:", torch.unique(X))
            #exit()
            image = X[j] #round()        # Take channel 0 and round (to 0 or 1)
            flattened = image.view(-1)       # Flatten the image to a 1D tensor
            # print the uniques in image 
            #print("Image unique values:", torch.unique(image))
            #exit()
            # count number of 0s and 1s
            #num_ones = torch.sum(flattened)
            #num_zeros = flattened.numel() - num_ones
            #print("Num ones:", num_ones)
            constraints = []
            for idx, pixel in enumerate(flattened):
                # Convert pixel to bool (0 becomes False, 1 becomes True)
                
                #print(pixel)
                pixel_bool = bool(pixel.item())
                constraints.append(input_expression[idx] == pixel_bool)
            # Combine constraints with And to form one big constraint.
            big_constraint = And(*constraints)
            batch_encodings.append(big_constraint)

        # Now iterate over the encodings of the batch.
        start = time.time()
        first = True
        #print(len(batch_encodings))
        for j, big_constraint in tqdm(enumerate(batch_encodings)):
            # Push the current state.
            local_robustness_solver.push()
            # Add the constraint for this image.
            local_robustness_solver.add(big_constraint)
            # Check the solver.
            z3_result = local_robustness_solver.check()
            # Record the result.
            if z3_result == sat:
                #print(f"Image {i} in batch is SAT: Local robustness violated!")
                results_array.append(False)
                print("Sat")
                #new_samples.append()
                model_counterexample = local_robustness_solver.model()
                x, xp = parse_ce_gc(model_counterexample, image_name='local_robustness', with_xp=True, verbose=False)
                # add the xp to the new samples and the label it should be
                # make it 2d 28,28
                #print("xp shape:", xp.shape)
                #xp = xp.view((1, 1, 28, 28))
                new_samples.append((xp, y[j]))
                
                #if first:
                    # run this if you want to see the first counterexample
                #    model_counterexample = local_robustness_solver.model()
                #    x, xp = parse_ce_gc(model_counterexample, image_name='local_robustness', with_xp=True)
                #    run_ce_trough_model(model, x)
                #    run_ce_trough_model(model, xp)
                #    first = False
                    
            else:
                #print(f"Image {i} in batch is UNSAT: Local robustness verified!")
                results_array.append(True)
            # Pop the input constraint so we return to the previous state.
            local_robustness_solver.pop()
        end = time.time()
        overall_sat_processing_time += end - start
        #if True:
        #    break
    # results array to torch tensor
    results_array = results_array
    #print("results:", results_array)
    print("number results" , len(results_array))
    #print("results:", print(results_array))
    # count the number of trues
    print("Number of Trues:", sum(results_array))
    
    print("Overall SAT processing time: ", overall_sat_processing_time, "s for ", len(results_array), " images")
    print(" Checking time per image: ", overall_sat_processing_time / len(results_array), "s")
    # verified_acc = compute_verified_accuracy(model, results_array, dataset, subset=True)
    #print(f"Verified accuracy on train set (eps={epsilon}): ", verified_acc)
    return new_samples, results_array

    
    
    
def sat_check_global_robustness(model, 
                                confidence, 
                                input_constraints_fn, 
                                relational_constraints,
                                constraints= None,
                                cnf=None,
                                kissat=False,
                                return_local_robustness_encoding=False,
                                x_assignment=None):
    
    input_size, N, C = get_model_infos(model)
    assert N % C == 0, "N must be divisible by C, this should also be enforced by the model"

    z3_input_expressions_x = vutils.init_z3_expressions_for_inputs(input_size, "x")
    z3_input_expressions_xp = vutils.init_z3_expressions_for_inputs(input_size, "xp")

    if x_assignment is not None:
        # this is used for the local robustness check
        input_assignment = vutils.encode_input_assignment(x_assignment, z3_input_expressions_x)

    input_constraints = input_constraints_fn(z3_input_expressions_x)
    input_constraints_xp = input_constraints_fn(z3_input_expressions_xp)

    all_relational_constraints = []
    print(f"Adding {len(relational_constraints)} relational constraints.")

    for const in relational_constraints:
        constraint = const(z3_input_expressions_x, z3_input_expressions_xp)
        all_relational_constraints.append(constraint)

    model_encoding_x, output_vars_x = vutils.encode_sat_model_with_vars_base_layers(model,
                                                                                    z3_input_expressions_x,
                                                                                    name="x")

    model_encoding_xp, output_vars_xp = vutils.encode_sat_model_with_vars_base_layers(model,
                                                                                    z3_input_expressions_xp,
                                                                                    name="xp")
    

    if confidence is not None:
        constraint_confidence = vutils.encode_confidence_largest_class_pure_sat(output_vars_x,
                                                                    N//C,
                                                                    C,
                                                                    kappa= confidence)

    constraint_class = vutils.encode_class_diff_general_pure_sat(output_vars_x,
                                                        output_vars_xp,
                                                        N = N//C,
                                                        C = C)

    s = z3.Solver()


    s.add(model_encoding_x["constraints"])
    s.add(model_encoding_xp["constraints"])

    s.add(input_constraints["constraints"])
    s.add(input_constraints_xp["constraints"])



    for rel in all_relational_constraints:
        s.add(rel["constraints"])
        
    #if skip_connections:
    #    print("Added skip connections!")
    #    s.add(relational_input_output_constraint["constraints"])
    
    s.add(constraint_class["constraints"])

    if return_local_robustness_encoding:
        return s, z3_input_expressions_x


    if x_assignment is not None:
        s.add(input_assignment["constraints"])
        s.add(constraint_confidence["constraints"])
        #print(input_assignment)
    else:
        s.add(constraint_confidence["constraints"])
    if constraints is not None:
        print("Adding additional constraints")
        print(constraints)
        raise NotImplementedError("Not implemented yet")

    temp_dir = os.path.join(tempfile.gettempdir(), "difflogic_verification")
    # print(cnf)
    if kissat and not cnf:
        os.makedirs(temp_dir, exist_ok=True)
        cnf = os.path.join(temp_dir, str(uuid.uuid1())) + ".cnf"
    if cnf:
        tseitin = z3.Tactic('tseitin-cnf')
        g = z3.Goal()
        g.add(s.assertions())
        normalized = tseitin(g).as_expr()
        g = z3.Goal()
        g.add(normalized)
        with open(cnf, "w") as f:
            f.write(g.dimacs())
        print("Wrote global robustness cnf to:", cnf)
        if not kissat:
            print("Exiting")
            exit()
        

    start = time.time()
    if kissat:
        result = subprocess.run(["kissat/build/kissat", "-q", cnf], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        kissat_output = result.stdout.strip()
        end_time = time.time() - start
        print("Time (SAT GR):", end_time)
        if cnf and cnf.startswith(temp_dir):
            os.remove(cnf)
        if "s UNSATISFIABLE" in kissat_output:
            return True, None, end_time
        else:
            return False, None, end_time
    else:
        result = s.check()
        end_time = time.time() - start
        print("Time (SAT GR):", end_time)
        if result == z3.unsat:
            return True, None, end_time
        else:
            m = s.model()
            return False, m, end_time
    