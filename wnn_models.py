import torch
import torch.nn as nn
import torch_dwn as dwn
import torch.nn.functional as F


def pad_dim_if_needed(dim, group_size):
    """Pad scalar dim so it's divisible by group_size."""
    #print(f"Padding dim {dim} to be divisible by group size {group_size}")
    r = dim % group_size
    if r == 0:
        return dim
    return dim + (group_size - r)

class RegressionBucketLayer(nn.Module):
    """Input: Popcounts for each dimension, Outputs: rescaled actions"""
    def __init__(self, n, k,
                 init_log_alpha=-0.6931): 
        super().__init__()
        self.n = float(n)
        self.k = float(k)
        
        init = torch.ones(k, dtype=torch.float32) *init_log_alpha
        self.log_alpha = nn.Parameter(init)
        self.beta = nn.Parameter(torch.zeros(k, dtype=torch.float32))
        self.eps = 1e-6
        self.norm_factor = self.n / self.k
    
    def forward(self, x):
        x_norm = x / self.norm_factor    
        # the clamp could be removed, its legacy, 
        # but the runs in the paper where done with it
        # keeping it here for consistency
        x_norm = torch.clamp(x_norm, self.eps, 1 - self.eps)
        y = torch.exp(self.log_alpha) *  (x_norm - 0.5) + self.beta
        return y
    


class WNN(nn.Module):
    def __init__(self, obs_dim, act_dim, bits,
                 thermometer, sizes=(1200, 1200), n=6,
                 device="cuda",
                 map="learnable", # the first layers connectivity
                 init_log_alpha=-0.6931,
                 later_learnable=True,
                 regression=True
                 ):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.bits = bits
        
        if len(sizes) < 2:
            base_out_dim = sizes[-1]                    
            padded_out_dim = pad_dim_if_needed(base_out_dim, act_dim)
            sizes[-1] = padded_out_dim
            
            actor_net_lgn = nn.Sequential(
                nn.Flatten(),
                dwn.LUTLayer(self.obs_dim* self.bits, 
                    padded_out_dim, n=n, 
                    mapping=map,)
            )
            
        else:
            actor_net_lgn = nn.Sequential(
                nn.Flatten(),
                dwn.LUTLayer(self.obs_dim* self.bits, 
                            sizes[0], n=n, 
                            mapping=map)
            )
            if later_learnable:
                map = "learnable"
            else:
                map = 'random'
            for i in range(1, len(sizes)-1):
                actor_net_lgn.append(dwn.LUTLayer(sizes[i - 1],
                                                sizes[i], 
                                                n=n,
                                                mapping=map,))
            base_out_dim = sizes[-1]                 
            padded_out_dim = pad_dim_if_needed(base_out_dim, act_dim) 
            sizes[-1] = padded_out_dim
            actor_net_lgn.append(dwn.LUTLayer(sizes[len(sizes)-2],
                                        sizes[len(sizes)-1], 
                                        n=n,
                                        mapping=map,))

            assert sizes[-1] % act_dim == 0, "Last LGN layer size must be divisible by act_dim"
            
        actor_net_lgn.append(dwn.GroupSum(k=act_dim, tau=1.0))
        if regression:
            actor_net_lgn.append(RegressionBucketLayer(sizes[-1], act_dim, 
                                                    init_log_alpha=init_log_alpha))

        self.thermometer = thermometer
        self.net = actor_net_lgn

    
        
    def forward(self, x):
        obs_t = torch.as_tensor(x, dtype=torch.float32)
        obs_bin = self.thermometer.binarize(obs_t)
        out = self.net(obs_bin)
        return out
    
    def set_training_mode(self, mode: bool = True):
        """
        Set the training mode for the network.
        """
        self.training = mode
        self.net.train(mode)
        return self