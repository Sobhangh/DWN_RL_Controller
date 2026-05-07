import torch
import numpy as np
import torch, torch.nn as nn
#@from torch.kmeans import kmeans        
from pathlib import Path
from typing import Union


class _ThermometerBoundaryGradFn(torch.autograd.Function):
    """Hard thermometer forward with custom boundary-based backward."""

    @staticmethod
    def forward(ctx, x, thresholds, cmp):
        # x: (..., D), thresholds: (D, B) or (B,), cmp: (..., D, B)
        # Note: this class did not exist in the initial version.
        ctx.save_for_backward(cmp)
        return cmp

    @staticmethod
    def backward(ctx, grad_output):
        (cmp,) = ctx.saved_tensors
        # grad_output: (..., D, B)
        n_bits = cmp.shape[-1]

        ones_count = cmp.sum(dim=-1).long()  # (..., D)

        grad_last_one = torch.zeros_like(ones_count, dtype=grad_output.dtype)
        has_last_one = ones_count > 0
        if has_last_one.any():
            last_idx = (ones_count - 1).clamp(min=0).unsqueeze(-1)
            gathered_last = torch.gather(grad_output, -1, last_idx).squeeze(-1)
            grad_last_one = torch.where(has_last_one, gathered_last, grad_last_one)

        grad_next_zero = torch.zeros_like(ones_count, dtype=grad_output.dtype)
        has_next_zero = ones_count < n_bits
        if has_next_zero.any():
            next_idx = ones_count.clamp(max=n_bits - 1).unsqueeze(-1)
            gathered_next = torch.gather(grad_output, -1, next_idx).squeeze(-1)
            grad_next_zero = torch.where(has_next_zero, gathered_next, grad_next_zero)

        # Rule:
        # 1) pass grad(last 1) if positive  → GD will push x down, flipping last 1→0 to reduce loss
        # 2) else pass grad(next 0) if negative → GD will push x up, flipping next 0→1 to reduce loss
        # 3) else pass 0
        use_last = grad_last_one > 0
        use_next = (~use_last) & (grad_next_zero < 0)
        grad_x = torch.where(
            use_last,
            grad_last_one,
            torch.where(use_next, grad_next_zero, torch.zeros_like(grad_next_zero)),
        )
        grad_x_mean = grad_output.abs().mean()
        grad_x = grad_x / (grad_x_mean)

        return grad_x , None, None #* n_bits * 100000

class ThermometerBase:
    """
    Base utilities shared by Thermometer + DistributiveThermometer.
    """
    def __init__(self, n_bits: int, feature_wise: bool = True, device ='cuda'):
        assert n_bits > 0
        self.n_bits      = int(n_bits)
        self.feature_wise  = bool(feature_wise)
        self.thresholds    = None          #  (..., num_bits) after fit()
        self.device = device

    # ------------- helper: to torch --------------------------
    @staticmethod
    def _as_tensor(x, device=None, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device) if device else x
        return torch.as_tensor(x, dtype=dtype, device=device)

    # ------------- main public API ---------------------------
    def fit(self, samples, *, min_value=None, max_value=None):
        """
        samples : (N, obs_dim) or (N,) NumPy / torch
        If min/max are provided they override data-driven bounds.
        """
        x = self._as_tensor(samples, device=self.device)

        if min_value is not None:
            # print(min_value, max_value)
            low  = self._as_tensor(min_value, device=self.device)
        else:
            low  = x.min(0)[0] if self.feature_wise else x.min()
        
        if max_value is not None:
            high = self._as_tensor(max_value, device=self.device)
        else:
            high = x.max(0)[0] if self.feature_wise else x.max()

        self.thresholds = self._compute_thresholds(x, low, high)  # (*dims, B)
        return self

    def encode(self, x, *, binary: bool = True, mid_point: bool = False):
        """
        x : (..., obs_dim) NumPy or torch
        binary=True  -> thermometer bits (..., obs_dim * B)
        binary=False -> discretised scalar  (..., obs_dim)
                        (mid_point=True   → mid point of bucket
                         mid_point=False  → bucket index 0..B)
        """
        # add a batch dimension to x if it doesn thave one
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if self.thresholds is None:
            raise RuntimeError("fit() must be called before encode().")

        x_t = self._as_tensor(x, device=self.thresholds.device)

        # make shape (..., obs_dim, 1) to broadcast thresholds (..., obs_dim, B)
        # print(self.thresholds.shape)
        cmp = (x_t.unsqueeze(-1) > self.thresholds).float()  # thermometer bits
        #print(cmp.shape)
        if binary:
            return cmp.flatten(-2)                            # (..., obs_dim*B)

        # discretised index: number of thresholds passed (0..B)
        bucket_idx = cmp.sum(-1)                              # (..., obs_dim)
        if mid_point:
            # map 0..B index → mid-point value between thresholds
            # edges:   -inf | t0 | t1 | ... | tB | +inf
            # choose mid = (left + right)/2
            thresholds = torch.cat(
                [self.thresholds[..., :1] - 1e-8,   # pad -inf approx
                 self.thresholds,
                 self.thresholds[..., -1:] + 1e-8], # pad +inf approx
                dim=-1
            ).unsqueeze(0) # add a batch dimension
            
            batch = bucket_idx.shape[0]
            thresholds = thresholds.expand(batch, -1, -1)
            left  = torch.gather(thresholds, -1, bucket_idx.long().unsqueeze(-1))
            right = torch.gather(thresholds, -1, bucket_idx.long().unsqueeze(-1) + 1)
            return (0.5 * (left + right)).squeeze(-1) 

        else:
            return bucket_idx          # integer bucket id

    # ------------- (override in subclass) --------------------
    def _compute_thresholds(self, x, low, high):
        raise NotImplementedError

    def cuda(self):
        self.thresholds = self.thresholds.cuda()
        print(self.thresholds.device)
    def binarize(self,x):
        # Initial version (no backward gradient through binarize):
        # with torch.no_grad():
        #     return self.encode(x, binary=True)

        if self.thresholds is None:
            raise RuntimeError("fit() must be called before binarize().")
        x_t = self._as_tensor(x, device=self.thresholds.device)
        if len(x_t.shape) == 1:
            x_t = x_t.unsqueeze(0)
        cmp_flat = self.encode(x_t, binary=True)
        cmp = cmp_flat.view(*x_t.shape, -1)
        cmp = _ThermometerBoundaryGradFn.apply(x_t, self.thresholds, cmp)
        return cmp.flatten(-2)


class ThermometerUniform(ThermometerBase):
    """
    Uniform (equal-width) buckets between `low` and `high`.
    If `feature_wise=True` you get one set of thresholds per feature
    -> shape (obs_dim, num_bits)
    Otherwise a global set shared by all dims -> shape (num_bits,)
    """
    def __init__(self, n_bits: int = 5, feature_wise: bool = True, device="cuda"):
        super().__init__(n_bits, feature_wise, device)

    # override -----------------------------------------------------------------
    def _compute_thresholds(self, x, low, high):
        """
        Return tensor of thresholds with last dim = num_bits
        """
        if self.feature_wise:
            delta = (high - low) / (self.n_bits + 1)
            idx   = torch.arange(1, self.n_bits + 1,
                                 device=x.device, dtype=x.dtype)
            thresholds = low.unsqueeze(-1) + idx * delta.unsqueeze(-1)  # (D,B)
        else:
            # scalar low/high  →  thresholds shared by all dims
            delta = (high - low) / (self.n_bits + 1)
            thresholds = low + torch.arange(
                1, self.n_bits + 1, device=x.device, dtype=x.dtype
            ) * delta                                             # (B,)
        return thresholds
    @property
    def codebook(self):
        return self.thresholds

    def binarize(self,x):
        # Initial version in this class (before delegating to custom-gradient base):
        # with torch.no_grad():
        #     return self.encode(x, binary=True)
        return super().binarize(x)

from torch.distributions.normal import Normal
class ThermometerGaussian(ThermometerBase):
    def __init__(self, n_bits: int = 5, feature_wise: bool = True, device="cuda"):
        super().__init__(n_bits, feature_wise, device)

    # ------------------------------------------------------------------
    #  Equal‑probability thresholds for a N(0,1)
    # ------------------------------------------------------------------
    def _compute_thresholds(self, x, low=None, high=None):
        """
        Create `n_bits` thresholds so that the k‑th threshold lies at the
        Gaussian quantile  k / (n_bits + 1).  Each adjacent pair therefore
        encloses 1/(n_bits + 1) of the probability mass.

        If `low` and `high` are supplied they are interpreted as the desired
        minimum and maximum threshold values; the quantiles are affinely
        rescaled to fit that range (useful when data are clipped).

        Returns
        -------
        thresholds : Tensor
            * shape (D, n_bits)   if feature_wise=True
            * shape (1, n_bits)   otherwise
        """
        assert low is not None and high is not None 
        device = x.device
        dtype  = x.dtype
        if x.ndim == 1:
            D = x.shape[0]  # (obs_dim,) → 1D
        else:
            D = x.shape[1]
        
        low    = self._as_tensor(low, device=device, dtype=dtype)
        high   = self._as_tensor(high, device=device, dtype=dtype)
        
        # Gaussian quantiles
        assert self.n_bits % 2 == 1, "n_bits must be odd for ThermometerGaussian"
        probs = torch.linspace(1, self.n_bits -1 , self.n_bits - 1,
                               device=device, dtype=dtype) / (self.n_bits)
        # add one at 0
        probs = torch.cat([torch.tensor([0.5], device=device, dtype=dtype), probs])
        probs, _ = torch.sort(probs)
        normal = Normal(0.0, 1.0)
        thr    = normal.icdf(probs)
        scale = torch.maximum(torch.abs(high), torch.abs(low)) /torch.max(thr)
        thr   = scale.unsqueeze(1) * thr.expand(D, -1)      
        if self.feature_wise:
            thr = thr
        else:
            thr = thr.unsqueeze(0)
        return thr
    
class DistributiveThermometer(ThermometerBase):
    """
    Thresholds so that each bucket has equal *count* of samples (per feature
    if feature_wise=True, else over the flattened distribution).
    """
    def __init__(self, n_bits: int = 1, feature_wise: bool = True, device="cuda"):
        super().__init__(n_bits, feature_wise, device=device)

    def _compute_thresholds(self, x, low, high):
        """
        Return thresholds tensor of shape (..., num_bits)
        where ... is () (global) or (obs_dim,) (feature-wise).
        """
        #print(x.shape)
        if self.feature_wise:
            # sort along batch dim, independently per feature
            data_sorted, _ = torch.sort(x, dim=0)
            n = data_sorted.shape[0]
            idx = (torch.arange(1, self.n_bits + 1, device=x.device)
                             * n // (self.n_bits + 1)).long()
            thresh = data_sorted[idx]                      # (B, obs_dim)
            # print(thresh.transpose(0, 1))
            
            return thresh.transpose(0, 1)                 # (obs_dim, B)
        else:
            data_sorted, _ = torch.sort(x.flatten())
            n = data_sorted.shape[0]
            idx = (torch.arange(1, self.n_bits + 1, device=x.device)
                             * n // (self.n_bits + 1)).long()
            return data_sorted[idx]                       # (B,) – broadcast OK
