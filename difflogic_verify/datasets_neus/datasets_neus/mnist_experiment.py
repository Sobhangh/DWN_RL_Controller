from utils import get_model, get_model_mnist_new
import torch

def build_mnist_model(layer_size, path=None, input_size=20*20, num_connections=64, last_layer=400):
    model = get_model(
                    grad_factor=1.0,
                    connections='learnable',
                    in_dim=input_size,
                    out_dim= 10, # as we need the standard deviation as well
                    k=layer_size,  # Adjust as needed
                    l=3,  # Adjust as needed
                    tau=10.0,
                    initialization='random', 
                    #bins=bins,
                    interconnect='gumbel_softmax',
                    discretization=None,
                    fill_to_one_hot=None,
                    use_packbits=False,
                    device='python',
                    freeze_first_layer=False,
                    num_connections = num_connections,
                    use_random_layer = True,
                    last_layer=last_layer
                )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    return model

def build_mnist_model_small(temp,C=10, path=None):
    model = get_model_mnist_new(
            grad_factor=1.0,
            connections='learnable',
            in_dim=20*20,
            out_dim= C, # as we need the standard deviation as well
            k=200,  # Adjust as needed
            l=3,  # Adjust as needed
            tau=10.0,
            initialization='random', 
            #bins=bins,
            interconnect='gumbel_softmax',
            discretization=None,
            fill_to_one_hot=None,
            use_packbits=False,
            device='python',
            freeze_first_layer=False,
            num_connections = 64,
            use_random_layer = True,
            last_layer=100
        )
    if path is not None:
        model.load_state_dict(torch.load(path, weights_only=False))
    return model