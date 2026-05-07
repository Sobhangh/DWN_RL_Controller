#!/usr/bin/env python3
"""
Utilities to load a PPO WNN actor checkpoint, harden LUT layers, and produce a Z3 encoding.

Design notes:
- Supports real-input encoding (obs norm + thermometer thresholds inside Z3).
- Also supports bit-input encoding directly (for debugging / constrained bit queries).
- Actor affine scaling (RegressionBucketLayer) is optional via include_affine.
- Works from a checkpoint directory containing:
    - ppo_train_docking.cleanrl_model
    - optionally obs_norm.pt
"""

import argparse
import contextlib
import io
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import z3
except ImportError as exc:
    raise ImportError("wnn_z3_encoder.py requires z3-solver in the active environment.") from exc

from thermometer import ThermometerGaussian


class _IdentityThermometer:
    """
    Helper for reference WNN loading: assumes caller provides already-binarized bits.
    """

    def binarize(self, x):
        return torch.as_tensor(x, dtype=torch.float32)


class _FixedThresholdThermometer:
    """
    Runtime thermometer wrapper for explicitly provided thresholds.
    """

    def __init__(self, thresholds: torch.Tensor):
        if thresholds.ndim != 2:
            raise ValueError(f"Expected thresholds shape [obs_dim,bits], got {tuple(thresholds.shape)}")
        self.thresholds = thresholds.detach().clone().to(torch.float32)

    def binarize(self, x):
        x_t = torch.as_tensor(x, dtype=torch.float32)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        return (x_t.unsqueeze(-1) > self.thresholds.to(x_t.device)).to(torch.float32).flatten(-2)


def convert_to_logic_gate(luts: torch.Tensor) -> torch.Tensor:
    """
    Convert 2-input LUTs (shape [out_dim, 4]) to one-hot over 16 Boolean operators.
    """
    if luts.ndim != 2 or luts.shape[1] != 4:
        raise ValueError(f"convert_to_logic_gate expects shape (out_dim,4), got {tuple(luts.shape)}")

    canonical = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.int32,
        device=luts.device,
    )
    luts_bin = (luts > 0).to(torch.int32)
    match = (luts_bin.unsqueeze(1) == canonical.unsqueeze(0)).all(dim=2)
    return match.to(torch.float32)


def convert_mapping_to_selected_inputs(mapping: torch.Tensor) -> torch.Tensor:
    mapping = mapping.flatten()
    num_elements = mapping.numel()
    if num_elements % 2 != 0:
        raise ValueError("Mapping length must be even to form gate input pairs.")
    out_dim = num_elements // 2
    selected_inputs = torch.zeros((out_dim, 2), dtype=torch.long, device=mapping.device)
    selected_inputs[:, 0] = mapping[0::2]
    selected_inputs[:, 1] = mapping[1::2]
    return selected_inputs.long()


def gate_expression_16(op_id: int, a_expr, b_expr):
    if op_id == 0:
        return z3.BoolVal(False)
    if op_id == 1:
        return z3.And(a_expr, b_expr)
    if op_id == 2:
        return z3.And(a_expr, z3.Not(b_expr))
    if op_id == 3:
        return a_expr
    if op_id == 4:
        return z3.And(b_expr, z3.Not(a_expr))
    if op_id == 5:
        return b_expr
    if op_id == 6:
        return z3.Xor(a_expr, b_expr)
    if op_id == 7:
        return z3.Or(a_expr, b_expr)
    if op_id == 8:
        return z3.Not(z3.Or(a_expr, b_expr))
    if op_id == 9:
        return z3.Not(z3.Xor(a_expr, b_expr))
    if op_id == 10:
        return z3.Not(b_expr)
    if op_id == 11:
        return z3.Or(z3.Not(b_expr), a_expr)
    if op_id == 12:
        return z3.Not(a_expr)
    if op_id == 13:
        return z3.Or(z3.Not(a_expr), b_expr)
    if op_id == 14:
        return z3.Not(z3.And(a_expr, b_expr))
    if op_id == 15:
        return z3.BoolVal(True)
    raise ValueError(f"Unsupported 16-gate op_id={op_id}")


class BaseLogicLayer(nn.Module):
    """
    Interface layer for hardened 2-input logic-gate networks.
    """

    def __init__(self, in_dim: int, out_dim: int, gate_set: int = 16):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.gate_set = int(gate_set)
        self.register_buffer("selected_inputs", torch.zeros((out_dim, 2), dtype=torch.long))
        self.weights = nn.Parameter(torch.zeros((out_dim, gate_set), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_set != 16:
            raise NotImplementedError("BaseLogicLayer.forward currently implemented for gate_set=16 only.")
        x_bin = (x > 0).to(torch.float32)
        a = x_bin[:, self.selected_inputs[:, 0]]
        b = x_bin[:, self.selected_inputs[:, 1]]
        op_ids = self.weights.argmax(dim=-1)

        out = torch.zeros_like(a)
        for i in range(self.out_dim):
            op = int(op_ids[i].item())
            ai = a[:, i] > 0.5
            bi = b[:, i] > 0.5
            if op == 0:
                oi = torch.zeros_like(ai)
            elif op == 1:
                oi = ai & bi
            elif op == 2:
                oi = ai & (~bi)
            elif op == 3:
                oi = ai
            elif op == 4:
                oi = bi & (~ai)
            elif op == 5:
                oi = bi
            elif op == 6:
                oi = torch.logical_xor(ai, bi)
            elif op == 7:
                oi = ai | bi
            elif op == 8:
                oi = ~(ai | bi)
            elif op == 9:
                oi = ~torch.logical_xor(ai, bi)
            elif op == 10:
                oi = ~bi
            elif op == 11:
                oi = (~bi) | ai
            elif op == 12:
                oi = ~ai
            elif op == 13:
                oi = (~ai) | bi
            elif op == 14:
                oi = ~(ai & bi)
            elif op == 15:
                oi = torch.ones_like(ai)
            else:
                raise ValueError(f"Unsupported gate id: {op}")
            out[:, i] = oi.to(torch.float32)
        return out

    def forward_z3_logic(self, input_z3_expressions: Sequence[object]) -> List[object]:
        if self.gate_set != 16:
            raise NotImplementedError("BaseLogicLayer.forward_z3_logic currently implemented for gate_set=16 only.")
        out_exprs: List[object] = []
        op_ids = self.weights.argmax(dim=-1).detach().cpu().numpy().astype(int)
        a_indices = self.selected_inputs[:, 0].detach().cpu().numpy()
        b_indices = self.selected_inputs[:, 1].detach().cpu().numpy()
        for i in range(self.out_dim):
            a_expr = input_z3_expressions[int(a_indices[i])]
            b_expr = input_z3_expressions[int(b_indices[i])]
            out_exprs.append(gate_expression_16(int(op_ids[i]), a_expr, b_expr))
        return out_exprs


@dataclass
class HardenedLUTLayer:
    input_size: int
    output_size: int
    n: int
    selected_inputs: torch.Tensor  # [out_dim, n], long
    lut_binary: torch.Tensor  # [out_dim, 2**n], bool

    def to_base_logic_layer(self) -> BaseLogicLayer:
        if self.n != 2:
            raise ValueError(f"to_base_logic_layer requires n=2, got n={self.n}")
        lgns = convert_to_logic_gate(self.lut_binary.to(torch.float32))
        bbl = BaseLogicLayer(in_dim=int(self.input_size), out_dim=int(self.output_size), gate_set=16)
        with torch.no_grad():
            bbl.selected_inputs[:, 0] = self.selected_inputs[:, 1]
            bbl.selected_inputs[:, 1] = self.selected_inputs[:, 0]
            bbl.weights.copy_(lgns)
        return bbl

    def forward_bits(self, x_bits: torch.Tensor, little_endian: bool = True) -> torch.Tensor:
        """
        x_bits: [batch, input_size] binary tensor (0/1 float or bool)
        returns: [batch, output_size] binary float
        """
        x = (x_bits > 0).to(torch.long)
        idx_inputs = self.selected_inputs.to(x.device)
        gathered = x[:, idx_inputs]  # [B, out_dim, n]
        if little_endian:
            weights = (2 ** torch.arange(self.n, device=x.device, dtype=torch.long)).view(1, 1, self.n)
        else:
            weights = (2 ** torch.arange(self.n - 1, -1, -1, device=x.device, dtype=torch.long)).view(1, 1, self.n)
        lut_idx = (gathered * weights).sum(dim=-1).long()  # [B, out_dim]

        lut = self.lut_binary.to(x.device).long().unsqueeze(0).expand(x.shape[0], -1, -1)
        out = torch.gather(lut, dim=2, index=lut_idx.unsqueeze(-1)).squeeze(-1)
        return out.to(torch.float32)

    def forward_z3_logic(self, input_z3_exprs: Sequence[object], little_endian: bool = True) -> List[object]:
        out_exprs: List[object] = []
        sel = self.selected_inputs.detach().cpu().numpy()
        lut = self.lut_binary.detach().cpu().numpy()

        for gate_idx in range(self.output_size):
            gate_inputs = [input_z3_exprs[int(i)] for i in sel[gate_idx]]
            true_terms = []
            for lut_idx in range(1 << self.n):
                if not bool(lut[gate_idx, lut_idx]):
                    continue
                if little_endian:
                    bits = [((lut_idx >> j) & 1) for j in range(self.n)]
                else:
                    bits = [((lut_idx >> (self.n - 1 - j)) & 1) for j in range(self.n)]
                conj = [gate_inputs[j] if bits[j] == 1 else z3.Not(gate_inputs[j]) for j in range(self.n)]
                true_terms.append(z3.And(*conj) if len(conj) > 1 else conj[0])

            if len(true_terms) == 0:
                out_exprs.append(z3.BoolVal(False))
            elif len(true_terms) == (1 << self.n):
                out_exprs.append(z3.BoolVal(True))
            elif len(true_terms) == 1:
                out_exprs.append(true_terms[0])
            else:
                out_exprs.append(z3.Or(*true_terms))

        return out_exprs


class WNNZ3Encoder:
    """
    Load a saved PPO WNN actor checkpoint and expose hardened forward + Z3 encoding.

    Inputs for encoding are bit-level actor inputs (thermometer output), i.e.
    shape [input_dim_bits]. Raw-observation -> thermometer is intentionally skipped.
    """

    def __init__(
        self,
        model_path: str,
        include_affine: bool = False,
        actor_prefix: str = "actor_mean",
        little_endian_lut_index: bool = True,
    ):
        self.model_path = os.path.abspath(model_path)
        self.include_affine = bool(include_affine)
        self.actor_prefix = actor_prefix
        self.little_endian_lut_index = bool(little_endian_lut_index)

        self.model_file = self._resolve_model_file(self.model_path)
        self.obs_norm_file = self._resolve_obs_norm_file(self.model_path)
        self.raw_checkpoint = torch.load(self.model_file, map_location="cpu")
        self.state_dict, self.checkpoint_meta = self._extract_state_dict_and_meta(self.raw_checkpoint)
        if not isinstance(self.state_dict, dict):
            raise TypeError(f"Expected a state_dict checkpoint at {self.model_file}, got {type(self.state_dict)}")
        self.actor_prefix = self._resolve_actor_prefix(self.state_dict, self.actor_prefix)

        self.obs_norm_state = None
        if self.obs_norm_file and os.path.isfile(self.obs_norm_file):
            self.obs_norm_state = torch.load(self.obs_norm_file, map_location="cpu")

        self.ckpt_config = self.checkpoint_meta.get("config", {}) if isinstance(self.checkpoint_meta, dict) else {}
        self.ckpt_input_ranges = self.checkpoint_meta.get("input_ranges") if isinstance(self.checkpoint_meta, dict) else None
        self.ckpt_thermometer_thresholds = self.checkpoint_meta.get("thermo_thresholds") 
        #print("Checkpoint thermometer thresholds:", self.ckpt_thermometer_thresholds)
        self.ckpt_thermometer_type = (
            self.checkpoint_meta.get("thermometer_type") if isinstance(self.checkpoint_meta, dict) else None
        )
        self.ckpt_uses_obs_norm = (
            self.checkpoint_meta.get("uses_obs_norm") if isinstance(self.checkpoint_meta, dict) else None
        )
        self._finish_init_from_loaded_state()

    @classmethod
    def from_state_dict(
        cls,
        state_or_payload,
        include_affine: bool = False,
        actor_prefix: str = "actor_mean",
        little_endian_lut_index: bool = True,
        obs_norm_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "WNNZ3Encoder":
        """
        Build encoder directly from an in-memory state payload.

        `state_or_payload` can be:
        - plain state_dict
        - payload with {"state_dict": ..., ...metadata...}
        """
        self = cls.__new__(cls)
        self.model_path = "<in-memory>"
        self.include_affine = bool(include_affine)
        self.actor_prefix = actor_prefix
        self.little_endian_lut_index = bool(little_endian_lut_index)
        self.model_file = "<in-memory>"
        self.obs_norm_file = None

        if isinstance(state_or_payload, dict):
            raw = state_or_payload
        else:
            raw = {"state_dict": state_or_payload}
        self.raw_checkpoint = raw
        self.state_dict, self.checkpoint_meta = cls._extract_state_dict_and_meta(raw)
        if not isinstance(self.state_dict, dict):
            raise TypeError(f"Expected a state_dict-like payload, got {type(self.state_dict)}")
        self.actor_prefix = self._resolve_actor_prefix(self.state_dict, self.actor_prefix)

        if obs_norm_state is not None:
            self.obs_norm_state = obs_norm_state
        else:
            self.obs_norm_state = None
            if isinstance(self.checkpoint_meta, dict):
                maybe_obs = self.checkpoint_meta.get("obs_norm_state")
                if isinstance(maybe_obs, dict):
                    self.obs_norm_state = maybe_obs

        self.ckpt_config = self.checkpoint_meta.get("config", {}) if isinstance(self.checkpoint_meta, dict) else {}
        self.ckpt_input_ranges = self.checkpoint_meta.get("input_ranges") if isinstance(self.checkpoint_meta, dict) else None
        self.ckpt_thermometer_thresholds = self.checkpoint_meta.get("thermo_thresholds")
        #print("Checkpoint thermometer thresholds:", self.ckpt_thermometer_thresholds)
        #exit()
        self.ckpt_thermometer_type = (
            self.checkpoint_meta.get("thermometer_type") if isinstance(self.checkpoint_meta, dict) else None
        )
        self.ckpt_uses_obs_norm = (
            self.checkpoint_meta.get("uses_obs_norm") if isinstance(self.checkpoint_meta, dict) else None
        )
        self._finish_init_from_loaded_state()
        return self

    def _finish_init_from_loaded_state(self) -> None:
        self.obs_dim, self.bits_per_obs = self._infer_obs_dim_and_bits()
        self.lut_layers: List[HardenedLUTLayer] = self._extract_hardened_lut_layers(self.state_dict , self.obs_dim * self.bits_per_obs)
        if not self.lut_layers:
            raise RuntimeError("No LUT layers found in actor checkpoint.")
        
        self.input_dim_bits = int(self.lut_layers[0].input_size)
        self.final_lut_dim = int(self.lut_layers[-1].output_size)

        self.group_k, self.group_tau = self._infer_group_sum(self.state_dict)
        self.group_width = int(self.final_lut_dim // self.group_k)

        self.reg_log_alpha, self.reg_beta = self._extract_regression_params(self.state_dict)
        self.has_regression = self.reg_log_alpha is not None and self.reg_beta is not None
        self._reference_wnn = None

        self._pre_thermo_clip_bounds: Optional[Tuple[float, float]] = (-10.0, 10.0)
        self._thermometer_runtime = self._build_runtime_thermometer()
        
        self.thermo_thresholds = (
            self._thermometer_runtime.thresholds.detach().cpu() if self._thermometer_runtime is not None else None
        )


    @staticmethod
    def _extract_state_dict_and_meta(raw_payload) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        if not isinstance(raw_payload, dict):
            raise TypeError(f"Expected dict checkpoint payload, got {type(raw_payload)}")

        meta = raw_payload
        if "state_dict" in raw_payload and isinstance(raw_payload["state_dict"], dict):
            return raw_payload["state_dict"], meta

        # Plain tensor-only state_dict format.
        if all(isinstance(v, torch.Tensor) for v in raw_payload.values()):
            return raw_payload, {}

        tensor_items = {k: v for k, v in raw_payload.items() if isinstance(v, torch.Tensor)}
        if tensor_items:
            return tensor_items, meta
        raise TypeError("Checkpoint dict does not contain a usable tensor state_dict.")

    @staticmethod
    def _resolve_actor_prefix(sd: Dict[str, torch.Tensor], requested_prefix: str) -> str:
        def has_luts(prefix: str) -> bool:
            pat = re.compile(rf"^{re.escape(prefix)}\.net\.(\d+)\.luts$")
            return any(pat.match(k) for k in sd.keys())

        candidates = [requested_prefix, "net", "actor_mean", "nn"]
        seen = set()
        for prefix in candidates:
            if prefix in seen:
                continue
            seen.add(prefix)
            if has_luts(prefix):
                return prefix

        # As a fallback, infer prefix from any '*.net.<idx>.luts' key.
        pat_any = re.compile(r"^(.+)\.net\.(\d+)\.luts$")
        for key in sd.keys():
            m = pat_any.match(key)
            if m:
                return m.group(1)
        return requested_prefix

    @staticmethod
    def _resolve_model_file(path: str) -> str:
        if os.path.isdir(path):
            candidate = os.path.join(path, "ppo_train_docking.cleanrl_model")
            if not os.path.isfile(candidate):
                raise FileNotFoundError(f"Could not find checkpoint file: {candidate}")
            return candidate
        if os.path.isfile(path):
            # Prefer adjacent state-dict checkpoint for script-saved module files
            # (e.g. torch.save(model, "foo.pt") with class under __main__).
            if path.endswith(".pt"):
                base, ext = os.path.splitext(path)
                if not base.endswith("_state_dict"):
                    sd_candidate = f"{base}_state_dict{ext}"
                    if os.path.isfile(sd_candidate):
                        return sd_candidate
            return path
        raise FileNotFoundError(path)

    @staticmethod
    def _resolve_obs_norm_file(path: str) -> Optional[str]:
        if os.path.isdir(path):
            candidate = os.path.join(path, "obs_norm.pt")
            return candidate if os.path.isfile(candidate) else None
        parent = os.path.dirname(path)
        candidate = os.path.join(parent, "obs_norm.pt")
        return candidate if os.path.isfile(candidate) else None

    def _state_key(self, suffix: str) -> Optional[str]:
        prefixed = f"{self.actor_prefix}.{suffix}"
        if prefixed in self.state_dict:
            return prefixed
        if suffix in self.state_dict:
            return suffix
        return None

    def _extract_input_ranges_from_meta(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        raw = self.ckpt_input_ranges
        if raw is None:
            return None
        try:
            pairs = [tuple(r) for r in raw]
            mins = torch.tensor([float(lo) for lo, _ in pairs], dtype=torch.float32)
            maxs = torch.tensor([float(hi) for _, hi in pairs], dtype=torch.float32)
        except Exception:
            return None
        if mins.ndim != 1 or maxs.ndim != 1 or mins.numel() == 0 or mins.numel() != maxs.numel():
            return None
        return mins, maxs

    def _infer_obs_dim_and_bits(self) -> Tuple[int, int]:
        thr = self.ckpt_thermometer_thresholds
        if thr is None:
            raise ValueError("Missing checkpoint thermometer thresholds; cannot infer obs_dim/bits.")
        thr_t = torch.as_tensor(thr)
        if thr_t.ndim != 2 or int(thr_t.shape[0]) <= 0 or int(thr_t.shape[1]) <= 0:
            raise ValueError(f"Invalid thermometer thresholds shape: {tuple(thr_t.shape)}")
        obs_dim = int(thr_t.shape[0])
        bits = int(thr_t.shape[1])
        return obs_dim, bits


    def _build_runtime_thermometer(self):
        if self.bits_per_obs <= 1:
            self._pre_thermo_clip_bounds = None
            return None

        # 1) Prefer exact checkpoint thresholds when present.
        thr = self.ckpt_thermometer_thresholds
        if thr is not None:
            thr_t = torch.as_tensor(thr)
            if thr_t.ndim == 2 and int(thr_t.shape[0]) > 0 and int(thr_t.shape[1]) > 0:
                # Exact thresholds path must not apply legacy pre-thermometer clipping.
                self._pre_thermo_clip_bounds = None
                return _FixedThresholdThermometer(thr_t)

        # 2) If checkpoint says uniform + provides ranges, rebuild uniform boundary thresholds.
        input_ranges = self._extract_input_ranges_from_meta()

        if input_ranges is not None and str(self.ckpt_thermometer_type).lower() == "uniform":
            mins, maxs = input_ranges
            if self.bits_per_obs <= 1:
                thr = mins.unsqueeze(1)
            else:
                idx = torch.arange(self.bits_per_obs, dtype=torch.float32)
                frac = idx / float(self.bits_per_obs - 1)
                thr = mins.unsqueeze(1) + frac.unsqueeze(0) * (maxs - mins).unsqueeze(1)
            self._pre_thermo_clip_bounds = None
            return _FixedThresholdThermometer(thr)

        # 3) Legacy PPO actor fallback: Gaussian thermometer on normalized/clipped [-10,10].
        thermo = ThermometerGaussian(n_bits=self.bits_per_obs, device="cpu")
        mins = torch.full((self.obs_dim,), -10.0, dtype=torch.float32)
        maxs = torch.full((self.obs_dim,), 10.0, dtype=torch.float32)
        thermo.fit(torch.zeros((1, self.obs_dim), dtype=torch.float32), min_value=mins, max_value=maxs)
        self._pre_thermo_clip_bounds = (-10.0, 10.0)
        return thermo

    def _extract_hardened_lut_layers(self, sd: Dict[str, torch.Tensor], input_size: int) -> List[HardenedLUTLayer]:
        pattern = re.compile(rf"^{re.escape(self.actor_prefix)}\.net\.(\d+)\.luts$")
        lut_indices = sorted(int(m.group(1)) for k in sd.keys() if (m := pattern.match(k)))

        layers: List[HardenedLUTLayer] = []
        for idx in lut_indices:
            luts = sd[f"{self.actor_prefix}.net.{idx}.luts"].detach().cpu()
            out_dim, lut_width = luts.shape
            n = int(round(math.log2(lut_width)))
            if (1 << n) != lut_width:
                raise ValueError(f"Layer {idx} LUT width is not power of 2: {lut_width}")

            map_weights_key = f"{self.actor_prefix}.net.{idx}.mapping.weights"
            map_tensor_key = f"{self.actor_prefix}.net.{idx}.mapping"
            if map_weights_key in sd:
                weights = sd[map_weights_key].detach().cpu()
                mapped = weights.argmax(dim=0).long()
                selected_inputs = mapped.view(out_dim, n)
                input_size = int(weights.shape[0])
            elif map_tensor_key in sd:
                mapping = sd[map_tensor_key].detach().cpu().long()
                if mapping.shape != (out_dim, n):
                    raise ValueError(
                        f"Layer {idx} mapping shape mismatch. expected {(out_dim, n)}, got {tuple(mapping.shape)}"
                    )
                selected_inputs = mapping
                input_size = input_size
            else:
                raise KeyError(f"Missing mapping for actor layer index {idx}")

            layers.append(
                HardenedLUTLayer(
                    input_size=input_size,
                    output_size=int(out_dim),
                    n=n,
                    selected_inputs=selected_inputs.long(),
                    lut_binary=(luts > 0),
                )
            )
        return layers

    def _infer_group_sum(self, sd: Dict[str, torch.Tensor]) -> tuple:
        if "actor_logstd" in sd:
            k = int(sd["actor_logstd"].shape[-1])
        else:
            _, beta = self._extract_regression_params(sd)
            if beta is None:
                raise ValueError("Unable to infer actor output dimension for GroupSum.")
            k = int(beta.shape[0])

        if self.final_lut_dim % k != 0:
            raise ValueError(f"Final LUT dim {self.final_lut_dim} not divisible by inferred act_dim {k}")
        return k, 1.0

    def _extract_regression_params(self, sd: Dict[str, torch.Tensor]) -> tuple:
        pattern_a = re.compile(rf"^{re.escape(self.actor_prefix)}\.net\.(\d+)\.log_alpha$")
        alpha_keys = [(int(m.group(1)), k) for k in sd.keys() if (m := pattern_a.match(k))]
        if not alpha_keys:
            return None, None
        alpha_keys.sort()
        idx, alpha_key = alpha_keys[-1]
        beta_key = f"{self.actor_prefix}.net.{idx}.beta"
        if beta_key not in sd:
            return None, None
        return sd[alpha_key].detach().cpu(), sd[beta_key].detach().cpu()

    def _normalized_interval(self, dim: int, lo: float, hi: float, include_obs_norm: bool = True) -> Tuple[float, float]:
        lo_f = float(lo)
        hi_f = float(hi)
        if lo_f > hi_f:
            lo_f, hi_f = hi_f, lo_f

        if include_obs_norm and self.obs_norm_state is not None and "mean" in self.obs_norm_state:
            mean = float(self.obs_norm_state["mean"][dim].item())
            var = float(self.obs_norm_state["var"][dim].item())
            std = math.sqrt(var + 1e-8)
            lo_f = (lo_f - mean) / std
            hi_f = (hi_f - mean) / std
            if lo_f > hi_f:
                lo_f, hi_f = hi_f, lo_f

        if self._pre_thermo_clip_bounds is not None:
            clip_lo, clip_hi = self._pre_thermo_clip_bounds
            lo_f = max(clip_lo, min(clip_hi, lo_f))
            hi_f = max(clip_lo, min(clip_hi, hi_f))
            if lo_f > hi_f:
                lo_f, hi_f = hi_f, lo_f
        return lo_f, hi_f

    def _region_to_bit_span(self, dim: int, lo: float, hi: float, include_obs_norm: bool, lower_open: bool) -> Dict[str, int]:
        if self.bits_per_obs <= 1:
            return {"min_ones": 0, "max_ones": 1, "start": 0, "end": 0}
        if self.thermo_thresholds is None:
            raise RuntimeError("Thermometer thresholds are unavailable.")

        nlo, nhi = self._normalized_interval(dim, lo, hi, include_obs_norm=include_obs_norm)
        thr = self.thermo_thresholds[dim]

        if lower_open:
            min_ones = int((nlo >= thr).sum().item())
        else:
            min_ones = int((nlo > thr).sum().item())
        max_ones = int((nhi > thr).sum().item())

        min_ones = max(0, min(self.bits_per_obs, min_ones))
        max_ones = max(0, min(self.bits_per_obs, max_ones))
        if max_ones < min_ones:
            max_ones = min_ones

        if max_ones == min_ones:
            start, end = -1, -1
        else:
            start = min_ones
            end = max_ones - 1
        return {"min_ones": min_ones, "max_ones": max_ones, "start": start, "end": end}

    def _normalized_z3_expr(self, real_expr, dim: int, include_obs_norm: bool = True):
        expr = real_expr
        if include_obs_norm and self.obs_norm_state is not None and "mean" in self.obs_norm_state:
            mean = float(self.obs_norm_state["mean"][dim].item())
            var = float(self.obs_norm_state["var"][dim].item())
            std = math.sqrt(var + 1e-8)
            expr = (expr - z3.RealVal(str(mean))) / z3.RealVal(str(std))
        if self._pre_thermo_clip_bounds is None:
            return expr
        lo = z3.RealVal(str(float(self._pre_thermo_clip_bounds[0])))
        hi = z3.RealVal(str(float(self._pre_thermo_clip_bounds[1])))
        return z3.If(expr < lo, lo, z3.If(expr > hi, hi, expr))

    def normalize_real_tensor(self, x_real: torch.Tensor, include_obs_norm: bool = True) -> torch.Tensor:
        if x_real.ndim == 1:
            x_real = x_real.unsqueeze(0)
        x = x_real.to(torch.float32)
        if include_obs_norm and self.obs_norm_state is not None and "mean" in self.obs_norm_state:
            mean = self.obs_norm_state["mean"].to(x.device, dtype=torch.float32)
            var = self.obs_norm_state["var"].to(x.device, dtype=torch.float32)
            x = (x - mean) / torch.sqrt(var + 1e-8)
        if self._pre_thermo_clip_bounds is None:
            return x
        return torch.clamp(x, float(self._pre_thermo_clip_bounds[0]), float(self._pre_thermo_clip_bounds[1]))

    def real_to_bits_tensor(self, x_real: torch.Tensor, include_obs_norm: bool = True) -> torch.Tensor:
        x = self.normalize_real_tensor(x_real, include_obs_norm=include_obs_norm)
        if self._thermometer_runtime is None:
            return (x > 0).to(torch.float32)
        return self._thermometer_runtime.binarize(x).to(torch.float32)

    def real_exprs_to_bit_exprs(
        self,
        real_input_exprs: Sequence[object],
        include_obs_norm: bool = True,
    ) -> List[object]:
        if len(real_input_exprs) != self.obs_dim:
            raise ValueError(f"Expected {self.obs_dim} real expressions, got {len(real_input_exprs)}")

        if self.bits_per_obs <= 1:
            return [real_input_exprs[i] > z3.RealVal("0.0") for i in range(self.obs_dim)]

        if self.thermo_thresholds is None:
            raise RuntimeError("Thermometer thresholds are unavailable.")

        bit_exprs: List[object] = []
        for dim in range(self.obs_dim):
            x_norm = self._normalized_z3_expr(real_input_exprs[dim], dim, include_obs_norm=include_obs_norm)
            thr = self.thermo_thresholds[dim]
            #print(len(thr))
            #print(self.bits_per_obs)
            for j in range(self.bits_per_obs):
                bit_exprs.append(x_norm > z3.RealVal(str(float(thr[j].item()))))
            #exit()
        return bit_exprs

    def to_base_logic_layers(self) -> List[BaseLogicLayer]:
        return [layer.to_base_logic_layer() for layer in self.lut_layers]

    def convert_real_inputs(
        self,
        input_region_of_shape_inputs: Sequence[Tuple[float, float]],
        include_obs_norm: bool = True,
        lower_open: bool = False,
    ) -> List[Tuple[int, int]]:
        """
        Over-approximate real input box -> thermometer bit ranges.

        Returns per raw input dimension a tuple (start, end) with the local bit-index
        interval that may vary. If no bit can vary for that dimension, returns (-1, -1).
        """
        if len(input_region_of_shape_inputs) != self.obs_dim:
            raise ValueError(f"Expected {self.obs_dim} input intervals, got {len(input_region_of_shape_inputs)}")

        out: List[Tuple[int, int]] = []
        for dim, (lo, hi) in enumerate(input_region_of_shape_inputs):
            span = self._region_to_bit_span(
                dim=dim,
                lo=float(lo),
                hi=float(hi),
                include_obs_norm=include_obs_norm,
                lower_open=lower_open,
            )
            out.append((int(span["start"]), int(span["end"])))
        return out

    # Alias for the typo spelling requested in the prompt.
    def conver_real_inputs(
        self,
        input_region_of_shape_inputs: Sequence[Tuple[float, float]],
        include_obs_norm: bool = True,
        lower_open: bool = False,
    ) -> List[Tuple[int, int]]:
        return self.convert_real_inputs(
            input_region_of_shape_inputs=input_region_of_shape_inputs,
            include_obs_norm=include_obs_norm,
            lower_open=lower_open,
        )

    def build_bit_constraints_from_real_region(
        self,
        bit_vars: Sequence[object],
        input_region_of_shape_inputs: Sequence[Tuple[float, float]],
        include_obs_norm: bool = True,
        lower_open: bool = False,
    ) -> List[object]:
        """
        Build Z3 constraints over thermometer bits from real-region overapproximation.
        Also enforces thermometer structure per dimension: a prefix of 1s followed by 0s.
        """
        if len(bit_vars) != self.input_dim_bits:
            raise ValueError(f"Expected {self.input_dim_bits} bit vars, got {len(bit_vars)}")
        if len(input_region_of_shape_inputs) != self.obs_dim:
            raise ValueError(f"Expected {self.obs_dim} input intervals, got {len(input_region_of_shape_inputs)}")

        constraints: List[object] = []
        for dim, (lo, hi) in enumerate(input_region_of_shape_inputs):
            span = self._region_to_bit_span(
                dim=dim,
                lo=float(lo),
                hi=float(hi),
                include_obs_norm=include_obs_norm,
                lower_open=lower_open,
            )
            min_ones = int(span["min_ones"])
            max_ones = int(span["max_ones"])
            base = dim * self.bits_per_obs

            # Thermometer consistency: b_j >= b_{j+1} (prefix ones, then zeros).
            for j in range(self.bits_per_obs - 1):
                curr = bit_vars[base + j]
                nxt = bit_vars[base + j + 1]
                constraints.append(z3.Implies(nxt, curr))

            for j in range(self.bits_per_obs):
                idx = base + j
                if j < min_ones:
                    constraints.append(bit_vars[idx])
                elif j >= max_ones:
                    constraints.append(z3.Not(bit_vars[idx]))
        return constraints

    def encode_z3_from_real_exprs(
        self,
        real_input_exprs: Sequence[object],
        include_affine: Optional[bool] = None,
        include_obs_norm: bool = True,
    ) -> List[object]:
        """
        Exact symbolic thermometer encoding from real expressions (with obs-norm).
        """
        return self.encode_z3(
            real_input_exprs,
            include_affine=include_affine,
            include_obs_norm=include_obs_norm,
            inputs_are_bits=False,
        )

    def _infer_reference_wnn_shape(self) -> Tuple[int, int, int, List[int], int]:
        act_dim = int(self.group_k)
        sizes = [int(layer.output_size) for layer in self.lut_layers]
        n = int(self.lut_layers[0].n)
        if any(layer.n != n for layer in self.lut_layers):
            raise ValueError("All LUT layers must have the same LUT arity n to build the reference WNN.")

        if self.obs_dim > 0 and self.bits_per_obs > 0 and self.obs_dim * self.bits_per_obs == self.input_dim_bits:
            return int(self.obs_dim), int(self.bits_per_obs), act_dim, sizes, n

        if self.obs_norm_state is not None and "mean" in self.obs_norm_state:
            raw_obs_dim = int(self.obs_norm_state["mean"].numel())
            if raw_obs_dim > 0 and self.input_dim_bits % raw_obs_dim == 0:
                bits = int(self.input_dim_bits // raw_obs_dim)
                return raw_obs_dim, bits, act_dim, sizes, n

        # Fallback: treat bit-vector input as direct observation with bits=1.
        return int(self.input_dim_bits), 1, act_dim, sizes, n

    def _actor_sub_state_dict(self) -> Dict[str, torch.Tensor]:
        prefix = f"{self.actor_prefix}."
        out: Dict[str, torch.Tensor] = {}
        for key, value in self.state_dict.items():
            if key.startswith(prefix):
                out[key[len(prefix) :]] = value
        return out

    def build_reference_wnn(self):
        """
        Build and load a standard WNN model (from wnn_models.WNN) for direct comparison.
        """
        if self._reference_wnn is not None:
            return self._reference_wnn

        from wnn_models import WNN

        obs_dim, bits, act_dim, sizes, n = self._infer_reference_wnn_shape()
        # WNN constructor prints architecture; suppress to keep CLI output focused.
        with contextlib.redirect_stdout(io.StringIO()):
            ref = WNN(
                obs_dim=obs_dim,
                act_dim=act_dim,
                bits=bits,
                thermometer=_IdentityThermometer(),
                sizes=sizes,
                n=n,
                device="cpu",
                map="learnable",
                init_log_alpha=-0.6931,
            )
        actor_sd = self._actor_sub_state_dict()
        missing, unexpected = ref.load_state_dict(actor_sd, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading reference WNN: {unexpected}")
        # Some torch versions return list[str], some tuple; both handled here.
        if missing:
            # Allow no omissions for actor path in this checkpoint format.
            raise RuntimeError(f"Missing keys when loading reference WNN: {missing}")

        ref.eval()
        self._reference_wnn = ref
        return ref

    @staticmethod
    def _find_group_sum_layer(model: nn.Module):
        for layer in model.net:
            if layer.__class__.__name__ == "GroupSum":
                return layer
        raise RuntimeError("Could not find GroupSum layer in reference WNN.")

    def _emulate_lut_layer_cpu(self, lut_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        CPU fallback for torch_dwn.LUTLayer forward.
        For binary inputs, this matches LUT lookup semantics used at inference.
        """
        mapping_obj = lut_layer.mapping
        if mapping_obj.__class__.__name__ == "LearnableMapping":
            x = mapping_obj(x)
            mapping = getattr(lut_layer, "_LUTLayer__dummy_mapping")
        else:
            mapping = mapping_obj

        mapping = mapping.to(x.device).long()  # [out_dim, n]
        x_bin = (x > 0.5).to(torch.long)
        gathered = x_bin[:, mapping]  # [B, out_dim, n]
        n = int(mapping.shape[1])

        if self.little_endian_lut_index:
            weights = (2 ** torch.arange(n, device=x.device, dtype=torch.long)).view(1, 1, n)
        else:
            weights = (2 ** torch.arange(n - 1, -1, -1, device=x.device, dtype=torch.long)).view(1, 1, n)

        lut_idx = (gathered * weights).sum(dim=-1).long()  # [B, out_dim]
        lut_values = lut_layer.luts.to(x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        out = torch.gather(lut_values, dim=2, index=lut_idx.unsqueeze(-1)).squeeze(-1)

        if getattr(lut_layer, "ste", True):
            out = (out > 0).to(torch.float32)
        return out

    def _forward_reference_cpu_emulated(self, ref: nn.Module, x_bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_bits
        pre_reg = None
        for layer in ref.net:
            name = layer.__class__.__name__
            if name == "Flatten":
                x = layer(x)
            elif name == "LUTLayer":
                x = self._emulate_lut_layer_cpu(layer, x)
            elif name == "GroupSum":
                x = layer(x)
                pre_reg = x.detach().clone()
            else:
                x = layer(x)
        if pre_reg is None:
            raise RuntimeError("Failed to capture pre-regression values from GroupSum hook.")
        return pre_reg, x.detach().clone()

    def forward_reference_wnn_bits(self, x_bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (pre_regression_group_sum, post_regression_output) from the reference WNN.
        """
        if x_bits.ndim == 1:
            x_bits = x_bits.unsqueeze(0)
        x_bits = (x_bits > 0).to(torch.float32)

        ref = self.build_reference_wnn()
        if torch.cuda.is_available():
            ref = ref.to("cuda")
            group_layer = self._find_group_sum_layer(ref)
            captures: Dict[str, torch.Tensor] = {}

            def _hook(_, __, output):
                captures["pre_reg"] = output.detach().cpu().clone()

            handle = group_layer.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    y_post = ref(x_bits.to("cuda")).detach().cpu().clone()
            finally:
                handle.remove()

            if "pre_reg" not in captures:
                raise RuntimeError("Failed to capture pre-regression values from GroupSum hook.")
            return captures["pre_reg"], y_post
        # CPU fallback: torch_dwn LUT kernels are CUDA-only, so emulate LUT forward.
        with torch.no_grad():
            return self._forward_reference_cpu_emulated(ref, x_bits)

    def forward_reference_wnn_real(
        self,
        x_real: torch.Tensor,
        include_obs_norm: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_bits = self.real_to_bits_tensor(x_real, include_obs_norm=include_obs_norm)
        return self.forward_reference_wnn_bits(x_bits)

    def forward_real(
        self,
        x_real: torch.Tensor,
        include_affine: Optional[bool] = None,
        include_obs_norm: bool = True,
    ) -> torch.Tensor:
        x_bits = self.real_to_bits_tensor(x_real, include_obs_norm=include_obs_norm)
        return self.forward_bits(x_bits, include_affine=include_affine)

    def forward_bits(self, x_bits: torch.Tensor, include_affine: Optional[bool] = None) -> torch.Tensor:
        """
        x_bits: [batch, input_dim_bits], binary input bits (thermometer output domain)
        returns actor output in the hardened approximation.
        """
        if x_bits.ndim == 1:
            x_bits = x_bits.unsqueeze(0)
        x = (x_bits > 0).to(torch.float32)
        if x.shape[-1] != self.input_dim_bits:
            raise ValueError(f"Expected input dim {self.input_dim_bits}, got {x.shape[-1]}")

        for layer in self.lut_layers:
            x = layer.forward_bits(x, little_endian=self.little_endian_lut_index)

        x = x.view(x.shape[0], self.group_k, self.group_width).sum(dim=-1) / self.group_tau

        use_affine = self.include_affine if include_affine is None else bool(include_affine)
        if use_affine:
            if not self.has_regression:
                raise RuntimeError("Requested include_affine=True but no regression params were found.")
            norm_factor = float(self.final_lut_dim) / float(self.group_k)
            x_norm = x / norm_factor
            x_norm = torch.clamp(x_norm, 1e-6, 1.0 - 1e-6)
            alpha = torch.exp(self.reg_log_alpha).view(1, -1)
            beta = self.reg_beta.view(1, -1)
            x = alpha * (x_norm - 0.5) + beta
        return x

    @staticmethod
    def _bool_to_real(expr_bool):
        return z3.If(expr_bool, z3.RealVal("1"), z3.RealVal("0"))

    def _encode_z3_from_bits(self, input_bit_exprs: Sequence[object], include_affine: Optional[bool] = None) -> List[object]:
        if len(input_bit_exprs) != self.input_dim_bits:
            raise ValueError(f"Expected {self.input_dim_bits} bit expressions, got {len(input_bit_exprs)}")

        curr = list(input_bit_exprs)
        for layer in self.lut_layers:
            curr = layer.forward_z3_logic(curr, little_endian=self.little_endian_lut_index)

        curr_real = [self._bool_to_real(e) for e in curr]

        grouped: List[object] = []
        for g in range(self.group_k):
            start = g * self.group_width
            end = (g + 1) * self.group_width
            grouped.append(z3.Sum(curr_real[start:end]) / z3.RealVal(str(float(self.group_tau))))

        use_affine = self.include_affine if include_affine is None else bool(include_affine)
        if not use_affine:
            return grouped

        if not self.has_regression:
            raise RuntimeError("Requested include_affine=True but no regression params were found.")

        norm_factor = float(self.final_lut_dim) / float(self.group_k)
        eps = 1e-6
        out_exprs: List[object] = []
        for i, expr in enumerate(grouped):
            x_norm = expr / z3.RealVal(str(norm_factor))
            x_clamped = z3.If(
                x_norm < z3.RealVal(str(eps)),
                z3.RealVal(str(eps)),
                z3.If(x_norm > z3.RealVal(str(1.0 - eps)), z3.RealVal(str(1.0 - eps)), x_norm),
            )
            alpha = float(torch.exp(self.reg_log_alpha[i]).item())
            beta = float(self.reg_beta[i].item())
            out_exprs.append(z3.RealVal(str(alpha)) * (x_clamped - z3.RealVal("0.5")) + z3.RealVal(str(beta)))
        return out_exprs

    def encode_z3(
        self,
        input_exprs: Sequence[object],
        include_affine: Optional[bool] = None,
        include_obs_norm: bool = True,
        inputs_are_bits: bool = False,
    ) -> List[object]:
        """
        Encode actor in Z3.

        Default mode expects real-valued inputs (length = obs_dim), applies optional
        obs normalization + clamp + thermometer thresholding internally, then propagates
        bits through the WNN logic and returns real-valued action expressions.

        Set inputs_are_bits=True to pass bit expressions directly (length = input_dim_bits).
        """
        if inputs_are_bits:
            bit_exprs = list(input_exprs)
        else:
            bit_exprs = self.real_exprs_to_bit_exprs(input_exprs, include_obs_norm=include_obs_norm)
        return self._encode_z3_from_bits(bit_exprs, include_affine=include_affine)


def _model_value_to_float(v) -> float:
    if z3.is_rational_value(v):
        return v.numerator_as_long() / v.denominator_as_long()
    s = v.as_decimal(20)
    if s.endswith("?"):
        s = s[:-1]
    return float(s)


def main() -> None:
    p = argparse.ArgumentParser(description="Load a PPO WNN actor checkpoint and compare real-input inference vs Z3 encoding.")
    p.add_argument(
        "--model-path",
        type=str,
        default="/nfs/scistore16/tomgrp/fkresse/difflogic/closed_loop/models_ppo/Docking2d__ppo_train_docking_wnn__1__1774945942",
    )
    p.add_argument("--include-affine", action="store_true", help="Include RegressionBucket affine layer in output.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--input-mode", type=str, choices=["random", "zeros", "ones"], default="random")
    p.add_argument("--input-scale", type=float, default=2.0, help="Scale for random real inputs in [-scale, scale].")
    p.add_argument("--disable-obs-norm", action="store_true", help="Disable ObsNorm in both reference and encoding.")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    encoder = WNNZ3Encoder(
        model_path=args.model_path,
        include_affine=args.include_affine,
        actor_prefix="actor_mean",
        little_endian_lut_index=True,
    )

    include_obs_norm = not args.disable_obs_norm

    if args.input_mode == "zeros":
        x_real = torch.zeros((1, encoder.obs_dim), dtype=torch.float32)
    elif args.input_mode == "ones":
        x_real = torch.ones((1, encoder.obs_dim), dtype=torch.float32)
    else:
        x_real = (2.0 * torch.rand((1, encoder.obs_dim), dtype=torch.float32) - 1.0) * float(args.input_scale)

    y_forward = encoder.forward_real(
        x_real,
        include_affine=args.include_affine,
        include_obs_norm=include_obs_norm,
    ).detach().cpu().numpy()[0]
    y_enc_pre = encoder.forward_real(
        x_real,
        include_affine=False,
        include_obs_norm=include_obs_norm,
    ).detach().cpu().numpy()[0]
    y_enc_post = (
        encoder.forward_real(
            x_real,
            include_affine=True,
            include_obs_norm=include_obs_norm,
        ).detach().cpu().numpy()[0]
        if encoder.has_regression
        else y_enc_pre
    )

    y_ref_pre_t, y_ref_post_t = encoder.forward_reference_wnn_real(
        x_real,
        include_obs_norm=include_obs_norm,
    )
    y_ref_pre = y_ref_pre_t.detach().cpu().numpy()[0]
    y_ref_post = y_ref_post_t.detach().cpu().numpy()[0]

    ref_pre_match = np.allclose(y_ref_pre.astype(np.float64), y_enc_pre.astype(np.float64), atol=1e-9, rtol=1e-9)
    ref_post_match = np.allclose(y_ref_post.astype(np.float64), y_enc_post.astype(np.float64), atol=1e-9, rtol=1e-9)

    real_vars = [z3.Real(f"x{i}") for i in range(encoder.obs_dim)]
    out_exprs = encoder.encode_z3(
        real_vars,
        include_affine=args.include_affine,
        include_obs_norm=include_obs_norm,
        inputs_are_bits=False,
    )

    s = z3.Solver()
    for i in range(encoder.obs_dim):
        s.add(real_vars[i] == z3.RealVal(str(float(x_real[0, i].item()))))

    out_vars = [z3.Real(f"out{i}") for i in range(len(out_exprs))]
    for i, expr in enumerate(out_exprs):
        s.add(out_vars[i] == expr)

    if s.check() != z3.sat:
        raise RuntimeError("Unexpected UNSAT while evaluating fixed input assignment.")

    m = s.model()
    y_z3 = np.array([_model_value_to_float(m.eval(v, model_completion=True)) for v in out_vars], dtype=np.float64)

    ok = np.allclose(y_forward.astype(np.float64), y_z3, atol=1e-9, rtol=1e-9)

    print("=== WNN Z3 Encoding Check ===")
    print(f"model_file: {encoder.model_file}")
    print(f"obs_norm_file: {encoder.obs_norm_file}")
    print(f"num_lut_layers: {len(encoder.lut_layers)}")
    print(f"obs_dim: {encoder.obs_dim}")
    print(f"input_dim_bits: {encoder.input_dim_bits}")
    print(f"group_k: {encoder.group_k}, group_width: {encoder.group_width}")
    print(f"include_affine: {args.include_affine}")
    print(f"include_obs_norm: {include_obs_norm}")
    print(f"input_mode: {args.input_mode}")
    print("x_real:", x_real[0].tolist())
    print("reference_pre_reg:", y_ref_pre.tolist())
    print("encoder_pre_reg:", y_enc_pre.tolist())
    print("reference_pre_reg_match:", ref_pre_match)
    print("reference_post_reg:", y_ref_post.tolist())
    print("encoder_post_reg:", y_enc_post.tolist())
    print("reference_post_reg_match:", ref_post_match)
    print("forward_output:", y_forward.tolist())
    print("z3_output:", y_z3.tolist())
    print("match:", ok)


if __name__ == "__main__":
    main()
