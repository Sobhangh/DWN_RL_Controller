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


from utils.sat_encoding import sat_check_model_existence, sat_check_global_robustness
import torch
from tqdm import tqdm
import argparse
from utils.exp_setup import setup_experiment
from utils.utils import eval_model, run_ce_trough_model
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

def pretty_print_stage(stage_num: int, message: str) -> None:
    # Print a rule with the stage name
    console.print(Rule(f"[bold cyan]Stage {stage_num}[/]"))
    # Print the message inside a styled panel
    console.print(Panel(message, style="bold white on dark_green", expand=False))
    # Close with another rule
    console.print(Rule())

def pretty_print_result(is_robust: bool, confidence: float, eps: float, sensitive_attribute:str) -> None:
    title = "RESULT"
    message = (
        f"Globally robust with confidence {confidence} (ε={eps}), sensitive attribute: {sensitive_attribute}"
        if is_robust
        else f"NOT globally robust with confidence {confidence} (ε={eps}), sensitive attribute: {sensitive_attribute}"
    )
    panel_style = "bold white on dark_green" if is_robust else "bold white on dark_red"
    console.print(Rule(title, style="cyan"))
    console.print(Panel(message, style=panel_style, expand=False))
    console.print(Rule())
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run verification experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="german_credit",
        help="Which experiment to run (e.g. 'german_credit', 'compas')."
    )
    parser.add_argument(
        "--verify",
        type=str,
        default="test_acc",
        help="Verification target (e.g. 'test_acc', 'global_robustness')."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence level for global robustness check."
    )
    parser.add_argument(
        "--epsilon",
        type=int,
        default=0,
        help="Epsilon for global robustness check."
    )
    parser.add_argument(
        "--epsilon_sensitive",
        type=int,
        default=1,
        help="Epsilon for the sensitive attirbute for global robustness check."
    )
    parser.add_argument(
        "--sensitive_attribute",
        type=str,
        default='None',
        help="Sensitive attribute."
    )
    parser.add_argument(
        "--usesmt",
        action="store_true",
        help="Use SMT solver for verification."
    )
    parser.add_argument(
        "--cnf",
        type=str,
        default=None,
        help="If provided write the formula to DIMACS cnf instead of solving it."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed.'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=100,
        help='Number of neurons.'
    )
    
    return parser.parse_args()


def verify_local_robustness(model, input_size, input_constraints_fn, 
                             relational_constraints, parse_ce_gc,
                             confidence=0.5, pure_sat=True,
                             cnf=None, kissat=False,
                             data=None,
                             subset=True):
    """
    Mainly for debugging and ensuring that GR is actually correct!
    """
    print("Starting local robustness check")
    if data is None:
        print("No data provided, please provide data for local robustness check")
        exit()
    
    train = data[0]
    test = data[1]
    print("Checking local robustness on train set")
    # we can just do sat_verify_global_robustness, but we add a constraint that x looks like the input image

    results_array = []   # This could be a list of tuples: (result, model or None)

    # Process each batch in the training set.
    sat_processing_times = []
    for batch in tqdm(train, desc="Processing batches"):
        X, y = batch  # X shape assumed: [batch_size, channels, height, width]
        batch_size = X.size(0)
        for i in range(batch_size):
            x = X[i]
            run_ce_trough_model(model, x)
            # check if the minimum confidence is satisfied, else skip
            gr,m,time  = sat_check_global_robustness(model, 
                                            confidence,
                                            input_constraints_fn, 
                                            relational_constraints,
                                            cnf=cnf, kissat=kissat,
                                            x_assignment= x
                                            )
            
            if not(gr):
                x, xp = parse_ce_gc(m, with_xp=True)
            results_array.append(gr)
            sat_processing_times.append(time)
            
            if i == 10:
                print("Stopping after 10 checks, because this is not efficiently implemented. As the solver is completely re-initalized, the checks themself are fast.")
                break
        if subset:
            break
    # results array to torch tensor
    results_array = torch.tensor(results_array)
    print("results:", results_array)
    print("Overall SAT processing time: ", sum(sat_processing_times), "s for ", len(results_array), " images")
    print(" Checking time per image: ", sum(sat_processing_times) / len(results_array), "s +- ", np.std(np.array(sat_processing_times)) )

def verify_global_robustness(experiment, model, input_constraints_fn, 
                             relational_constraints, parse_ce_gc, N, C,
                             confidence=0.5, epsilon=0, pure_sat=True,
                             cnf=None, kissat=False,
                             sensitive_attribute=None,
                             ):
    """
    @brief Check if the model is globally robust with a certain confidence level
    """
    # check if such a model exists
    
    pretty_print_stage(1, "Checking if valid inputs exist that satisfy the conditions")
    exists, m, time_to_check_existence = sat_check_model_existence(model,
                                                confidence,
                                                input_constraints_fn, 
                                                #N, C,
                                                )
    print("Valid inputs that satisfy the conditions exist: ", exists, " (confidence: ", confidence, ")")
    if exists:
        print("Example input:")
        x, xp = parse_ce_gc(m, with_xp=False)
        run_ce_trough_model(model, x) # sanity ch
    else:
        print("No valid inputs that satisfy the conditions exist ...")
    pretty_print_stage(2, "Starting global robustness check")    
    results = sat_check_global_robustness(model, 
                                        confidence,
                                        input_constraints_fn, 
                                        relational_constraints,
                                        cnf=cnf, kissat=kissat)
    robustness, m, time_to_check_robustness = results
    
    if robustness:
        pretty_print_result(True, confidence=confidence, 
                            eps=epsilon, sensitive_attribute=sensitive_attribute)
    
        #print(f"Globally robust with confidence {confidence} (eps={epsilon})")
        if not exists:
            print("!HOWEVER! No valid inputs that satisfy the confidence threshold exist.")
    else:
        
        if kissat:
            print("Cannot parse CE for kissat, if you want to parse it, please use --usesmt")
        else:
            if "mnist" in experiment:
                x, xp = parse_ce_gc(m, image_name='global_robustness', with_xp=True)
            else:
                x, xp = parse_ce_gc(m, with_xp=True)
            print("x as input")
            run_ce_trough_model(model, x)
            print("x' as input")
            run_ce_trough_model(model, xp)
        pretty_print_result(False, confidence=confidence, 
                            eps=epsilon, sensitive_attribute=sensitive_attribute)
        #print(f"NOT globally robust with confidence {confidence} (eps={epsilon})")
    return robustness, time_to_check_existence, time_to_check_robustness


def main():
    import json
    args = parse_args()  # Parse command-line args    
    pretty_print_stage(0, "Setting up datasets and model")
    (model, input_size, input_constraints_fn, relational_constraints, parse_ce_gc, N, C), data = setup_experiment(args)
    print("Evaluating model")
    print("make sure you use the same seed as for training, so the splits are the same! Important for accuracy to be correct")
    print(model)
    acc_test = eval_model(model, data[1], mode=False)
    print(f"Test accuracy: {acc_test}")
    
    if args.verify == 'test_acc':
        with open(f"result.json", "w") as f:
            json.dump({"accuracy": acc_test}, f)
                
    elif args.verify == "global_robustness":
        
        results = verify_global_robustness(args.experiment, 
                                 model, 
                                 input_constraints_fn, 
                                 relational_constraints, 
                                 parse_ce_gc,
                                 N, 
                                 C,
                                 confidence=args.confidence, 
                                 epsilon=args.epsilon,
                                 pure_sat=not(args.usesmt),
                                 cnf=args.cnf,
                                 kissat=not(args.usesmt),
                                 sensitive_attribute=args.sensitive_attribute,
                                 )
        
        robustness, time_to_exists, time_to_verify = results
        with open(f"result.json", "w") as f:
            json.dump({"robustness": robustness, "time_to_exists": time_to_exists, "time_to_verify": time_to_verify, "confidence": args.confidence, "accuracy": acc_test}, f)
            
    elif args.verify == "local_robustness":
        results = verify_local_robustness(model, 
                                 input_size, 
                                 input_constraints_fn, 
                                 relational_constraints, 
                                 parse_ce_gc,
                                 confidence=args.confidence, 
                                 pure_sat=not(args.usesmt),
                                 cnf=args.cnf,
                                 kissat=not(args.usesmt),
                                 data = data)
    else:
        raise ValueError("Unknown verification type")

if __name__ == "__main__":
    main()
