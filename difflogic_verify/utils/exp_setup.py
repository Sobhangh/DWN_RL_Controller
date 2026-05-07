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
    Setting up datasets and experiments.
"""

import utils.verification_utils as vutils
from .utils import *
import torch
from functools import partial
from datasets_neus import dataset

def build_model(in_dim, C=2, k=100, l=3, device='cpu'):
    layers = []
    layers.append(BaseLogicLayer(in_dim=in_dim, out_dim=k, 
                                 device=device, initalization='random'))
    for _ in range(l-1):
        layers.append(BaseLogicLayer(in_dim=k, out_dim=k, 
                                     device=device, initalization='random'))
    layers.append(BaseGroupSum(k=C,in_dim=k, tau=20.0, device=device))
    model = torch.nn.Sequential(*layers)
    return model

def setup_german_credit(args):
    loaders, categorical_info, numeric_info, C, in_dim = dataset.build_dataset(
        "german_credit", batch_size=128, seed=args.seed
    )

    model = build_model(in_dim, C, k=args.k, device="cpu")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    data = (loaders.train, loaders.test)
    C, input_size, N = 2, in_dim, model[-2].out_dim

    input_constraints_fn = partial(
        vutils.build_input_domain_constraints,
        categorical_info=categorical_info,
        numeric_info=numeric_info,
    )

    relational_constraints = []

    if not (args.usesmt):
        encode_temp = vutils.encode_temperature_constraints_pure_sat

    else:
        encode_temp = vutils.encode_temperature_constraints

    temp_constraint_fn = partial(
        encode_temp, numerical_info=numeric_info, epsilon=args.epsilon
    )

    if args.sensitive_attribute != "age":
        relational_constraints = [
            temp_constraint_fn,
            partial(
                vutils.encode_sensitive_attribute_flip,
                categorical_info=categorical_info,
                attribute_to_flip=args.sensitive_attribute,
            ),
        ]
    else:

        print(f"Sensitive attribute is age {args.epsilon_sensitive}")
        rest_numeric = []
        for info in numeric_info:
            if info["col"] == "age":
                numeric_age = info
            else:
                rest_numeric.append(info)
        numeric_info_age = [numeric_age]
        relational_constraints = [
            temp_constraint_fn,
            # enforce that no categorica attribute is flipped
            partial(
                vutils.encode_sensitive_attribute_flip,
                categorical_info=categorical_info,
                attribute_to_flip="None",
            ),
            partial(
                encode_temp,
                numerical_info=numeric_info_age,
                epsilon=args.epsilon_sensitive,
            ),
            partial(encode_temp, numerical_info=rest_numeric, epsilon=args.epsilon),
        ]

    parse_ce_gc = partial(
        parse_counterexample,
        numeric_dict=numeric_info,
        cat_dict=categorical_info,
        C=C,
        in_dim=input_size,
        out_dim=N,
    )

    #acc = eval_model(model, loaders.test, mode=False)
    #print("Test accuracy:", acc)
    return (
        model,
        input_size,
        input_constraints_fn,
        relational_constraints,
        parse_ce_gc,
        N,
        C,
    ), data


def setup_law(args):
    loaders, categorical_info, numeric_info, C, in_dim = dataset.build_dataset(
        "law", batch_size=128, seed=args.seed
    )

    model = build_model(
        in_dim,
        C,
        k=args.k,
    )
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    data = (loaders.train, loaders.test)
    C, input_size, N = 2, in_dim, model[-2].out_dim

    input_constraints_fn = partial(
        vutils.build_input_domain_constraints,
        categorical_info=categorical_info,
        numeric_info=numeric_info,
    )

    relational_constraints = []

    if args.usesmt:
        temp_constraint_fn = partial(
            vutils.encode_temperature_constraints,
            numerical_info=numeric_info,
            epsilon=args.epsilon,
        )
    else:
        temp_constraint_fn = partial(
            vutils.encode_temperature_constraints_pure_sat,
            numerical_info=numeric_info,
            epsilon=args.epsilon,
        )

    # if args.verify == "global_robustness":
    relational_constraints = [
        temp_constraint_fn,
        partial(
            vutils.encode_sensitive_attribute_flip,
            categorical_info=categorical_info,
            attribute_to_flip=args.sensitive_attribute,
        ),
    ]
    parse_ce_gc = partial(
        parse_counterexample,
        numeric_dict=numeric_info,
        cat_dict=categorical_info,
        C=C,
        in_dim=input_size,
        out_dim=N,
    )
    return (
        model,
        input_size,
        input_constraints_fn,
        relational_constraints,
        parse_ce_gc,
        N,
        C,
    ), data


def setup_adult(args):
    # from experiments.training import build_folklore_model
    # from experiments.training import build_adult_ds
    loaders, categorical_info, numeric_info, C, in_dim = dataset.build_dataset(
        "adult", batch_size=128, seed=args.seed
    )

    model = build_model(in_dim, C, k=args.k, device="cpu")
    # print(model, in_dim, C)
    # print(args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    data = (loaders.train, loaders.test)
    C, input_size, N = 2, in_dim, model[-2].out_dim

    input_constraints_fn = partial(
        vutils.build_input_domain_constraints,
        categorical_info=categorical_info,
        numeric_info=numeric_info,
    )

    relational_constraints = []
    if args.usesmt:
        temp_constraint_fn = partial(
            vutils.encode_temperature_constraints,
            numerical_info=numeric_info,
            epsilon=args.epsilon,
        )
    else:
        temp_constraint_fn = partial(
            vutils.encode_temperature_constraints_pure_sat,
            numerical_info=numeric_info,
            epsilon=args.epsilon,
        )

    # if args.verify == "global_robustness":
    relational_constraints = [
        temp_constraint_fn,
        partial(
            vutils.encode_sensitive_attribute_flip,
            categorical_info=categorical_info,
            attribute_to_flip=args.sensitive_attribute,
        ),
    ]
    parse_ce_gc = partial(
        parse_counterexample,
        numeric_dict=numeric_info,
        cat_dict=categorical_info,
        C=C,
        in_dim=input_size,
        out_dim=N,
    )
    return (
        model,
        input_size,
        input_constraints_fn,
        relational_constraints,
        parse_ce_gc,
        N,
        C,
    ), data


def setup_folklore_5(args):
    loaders, categorical_info, numeric_info, C, in_dim = dataset.build_dataset(
        "folktable_5", batch_size=128, seed=args.seed
    )

    model = build_model(in_dim, C, k=args.k, device="cpu")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # input_size, C, N = X_train.shape[1], 5, get_output_layer_size(model)
    input_constraints_fn = partial(
        vutils.build_input_domain_constraints,
        categorical_info=categorical_info,
        numeric_info=numeric_info,
    )
    relational_constraints = []

    # if args.verify == "global_robustness":

    if not (args.usesmt):
        temperature_constraint_fn = partial(
            vutils.encode_temperature_constraints_pure_sat,
            numerical_info=numeric_info,
            epsilon=args.epsilon,
        )
    else:
        temperature_constraint_fn = partial(
            vutils.encode_temperature_constraints,
            numerical_info=numeric_info,
            epsilon=args.epsilon,
        )

    sensitivity_constraint_fn = partial(
        vutils.encode_sensitive_attribute_flip,
        categorical_info=categorical_info,
        attribute_to_flip=args.sensitive_attribute,
    )
    relational_constraints = [temperature_constraint_fn, sensitivity_constraint_fn]
    parse_ce_gc = partial(
        parse_counterexample,
        numeric_dict=numeric_info,
        cat_dict=categorical_info,
        C=C,
        in_dim=in_dim,
        out_dim=model[-2].out_dim,
    )

    data = (loaders.train, loaders.test)
    #acc = eval_model(model, loaders.test, mode=False)
    #print("Test accuracy:", acc)

    return (
        model,
        in_dim,
        input_constraints_fn,
        relational_constraints,
        parse_ce_gc,
        model[-2].out_dim,
        C,
    ), data


def setup_compas(args):
    loaders, categorical_info, numeric_info, C, in_dim = dataset.build_dataset(
        "compas", batch_size=128, seed=args.seed
    )

    model = build_model(in_dim, C, k=args.k, device="cpu")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    #acc = eval_model(model, loaders.test, mode=False)
    #print("Test accuracy:", acc)

    input_size, C, N = in_dim, 3, model[-2].out_dim
    data = (loaders.train, loaders.test)

    input_constraints_fn = partial(
        vutils.build_input_domain_constraints,
        categorical_info=categorical_info,
        numeric_info=numeric_info,
    )
    relational_constraints = []

    if not (args.usesmt):
        temperature_constraint_fn = partial(
            vutils.encode_temperature_constraints_pure_sat,
            numerical_info=numeric_info,
            epsilon=args.epsilon,
        )
    else:
        temperature_constraint_fn = partial(
            vutils.encode_temperature_constraints,
            numerical_info=numeric_info,
            epsilon=args.epsilon,
        )

    sensitivity_constraint_fn = partial(
        vutils.encode_sensitive_attribute_flip,
        categorical_info=categorical_info,
        attribute_to_flip=args.sensitive_attribute,
    )
    relational_constraints = [temperature_constraint_fn, sensitivity_constraint_fn]
    parse_ce_gc = partial(
        parse_counterexample,
        numeric_dict=numeric_info,
        cat_dict=categorical_info,
        C=C,
        in_dim=input_size,
        out_dim=N,
    )
    return (
        model,
        input_size,
        input_constraints_fn,
        relational_constraints,
        parse_ce_gc,
        N,
        C,
    ), data


def setup_experiment(args):
    """
    Sets up the experiment by loading the appropriate model and constraints.
    Returns: model, input_size, input_constraints_fn, relational_constraints, parse_ce_gc, N, C
    """
    if args.experiment == "german_credit":
        infos, data = setup_german_credit(args)
    elif args.experiment == "compas":
        infos, data = setup_compas(args)
    elif "mnist20x20" in args.experiment:
        infos, data = setup_mnist_20(args)
    elif "mnist" in args.experiment:
        infos, data = setup_mnist(args)
    elif args.experiment == "adult":
        infos, data = setup_adult(args)
    elif args.experiment == "law" or args.experiment == "lawNoWeights":
        infos, data = setup_law(args)
    elif args.experiment == "folktable_5":
        infos, data = setup_folklore_5(args)
    else:
        raise ValueError("Unknown experiment")
    return infos, data
