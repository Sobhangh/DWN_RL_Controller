# import math
# import random
# from typing import Union

# import torch
# from torch.nn.common_types import _size_2_t, _size_3_t
# from torch.nn.modules.utils import _pair, _triple
# from .mapping import layer_mapping

# from ..connections import setup_connections
# from ..functional import (
#     get_regularization_loss, rescale_weights
#     )
# from .lut_layer import LUTLayer


# class _LogicConvNd(LUTLayer):
#     """Abstract baseclass for convolutional logic layers.
#     This module provides common functionality for 2D and 3D logic convolutional
#     layers with differentiable learning.
    
#     Args:
#         in_dim: Input spatial dimensions ``(depth, height, width)``.
#         channels: Number of input channels.
#         num_kernels: Number of output logic kernels (analogous to output channels)
#         tree_depth: Depth of the binary logic tree. A depth of ``d`` uses
#             ``2**d`` leaves per receptive field.
#         receptive_field_size: Spatial size (depth, height and width) of the
#             receptive field (assumed cubic).
#         stride: Convolution stride in all spatial dimensions.
#         padding: Zero-padding applied symmetrically to depth, height and width
#             before selecting receptive fields.
#         conv_dimension: Dimension of the convolution (2 or 3).
#         device (str): Device to run the layer on ('cpu' or 'cuda').
#         lut_rank (int): Rank of the LUTs used in the layer.
#         """

#     def __init__(
#         self,
#         in_dim: Union[_size_2_t, _size_3_t, int],
#         channels: int = 1,
#         num_kernels: int = 16,
#         tree_depth: int = None,
#         receptive_field_size: Union[_size_2_t, _size_3_t, int] = 2,
#         stride: int = 1,
#         padding: int = 0,
#         conv_dimension: int = 2,
#         device: str = "cpu",
#         lut_rank: int = 2,
#     ):
#         super().__init__(
#             n=lut_rank,
#             )
#         self.num_kernels = num_kernels
#         self.tree_depth = tree_depth
#         self.channels = channels
#         self.conv_dimension = conv_dimension
#         assert conv_dimension in [2, 3], "conv_dimension must be 2 or 3"
#         if conv_dimension == 2:
#             self.receptive_field_size = _pair(receptive_field_size)
#             self.in_dim = _pair(in_dim)
#         else:
#             self.receptive_field_size = _triple(receptive_field_size)
#             self.in_dim = _triple(in_dim)
#         assert (
#             all(stride <= dim for dim in self.receptive_field_size)
#         ), (
#             f"Stride ({stride}) cannot be larger than "
#             f"receptive field size ({receptive_field_size})"
#         )        
#         self.stride = stride
#         self.padding = padding
#         self.tree_weights = self._init_weights()
#         self.indices = self._init_connections()
#         #self.connections = self._init_connections()
        
#     def _init_weights(self):
#         # Initialize tree weights using parametrization
#         tree_weights = torch.nn.ParameterList()
#         for i in reversed(range(self.tree_depth)):
#             # each tree level has lut_rank**i nodes per kernel
#             level_weights = torch.nn.Parameter(torch.stack(
#                 [   torch.rand(self.num_kernels, 2**self.lut_rank, dtype=torch.float32)*2 - 1
#                     # self.parametrization.init_weights(
#                     #     self.num_kernels, 
#                     #     self.device
#                     # ) 
#                     for _ in range(self.lut_rank**i)
#                 ]
#             ))
#             tree_weights.append(level_weights)
#         return tree_weights

#     # def _init_connections(self):
#     #      # Setup connections
#     #     self.connections = setup_connections(
#     #         structure="conv",
#     #         connections=self.connections,
#     #         lut_rank=self.lut_rank,
#     #         device=self.device,
#     #         in_dim=self.in_dim,
#     #         channels=self.channels,
#     #         num_kernels=self.num_kernels,
#     #         tree_depth=self.tree_depth,
#     #         receptive_field_size=self.receptive_field_size,
#     #         conv_dimension=self.conv_dimension,
#     #         stride=self.stride,
#     #         padding=self.padding,
#     #         **self.connections_kwargs
#     #     )
#     #     return self.connections

#     def forward(self, x):
#         """Applies the logic convolution to the input.

#         The forward pass proceeds as follows:

#         1. Optionally pad the input spatially.
#         2. Select all receptive-field positions for the first tree level using
#            precomputed index tensors.
#         3. For each tree level:
#             a. Sample or select LUT weights for all nodes at that level.
#             b. Apply binary logic operations to the child activations,
#                reducing them up the tree.
#         4. Reshape the final per-kernel outputs into a 4D tensor of shape
#            ``(batch_size, num_kernels, out_height, out_width)``.

#         Args:
#             x: Input tensor of shape ``(batch_size, channels, height, width)``.

#         Returns:
#             Tensor of shape ``(batch_size, num_kernels, out_height, out_width)``,
#             where ``out_height`` and ``out_width`` are determined by the
#             convolution parameters:

#             * ``out_height = (in_height + 2 * padding - receptive_field_size) // stride + 1``
#             * ``out_width  = (in_width  + 2 * padding - receptive_field_size) // stride + 1``.
#         """
#         if self.padding > 0:
#             x = torch.nn.functional.pad(
#                 x,
#                 (self.padding, self.padding, self.padding, self.padding, 0, 0),
#                 mode="constant",
#                 value=0
#             )
#         # First level tree indices
#         x = self.connections(x, 0)
#         # Process first level with einsum contraction
#         # b=batch, c=channels, s=spatial, f=features, k=num_basis/16
#         x = self.parametrization.forward(
#             x, self.tree_weights[0], self.training,
#             contraction='fc,bcsf->bcsf'
#         )
#         # Process remaining levels
#         for level in range(1, self.tree_depth):
#             x = self.connections(x, level)
#             x = x.movedim(-2, 1)
#             x = self.parametrization.forward(
#                 x, self.tree_weights[level], self.training,
#                 contraction='fc,bcsf->bcsf'
#             )
#         # Reshape flattened output
#         reshape = [(in_dim + 2*self.padding - rfs) // self.stride + 1 
#                    for in_dim, rfs in zip(self.in_dim, self.receptive_field_size)]
#         x = x.view(x.shape[0], x.shape[1], *reshape)

#         return x

#     def get_luts_and_ids(self):
#         """Computes the most probable LUT and its ID for each neuron.

#         Returns:
#             Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
#                 - ``tree_luts``: Nested list of Boolean tensors (truth tables)
#                 - ``tree_ids``: Nested list of integer tensors (LUT IDs)
#         """
#         tree_ids = []
#         tree_luts = []
#         for level in range(self.tree_depth):
#             level_ids = []
#             level_luts = []
#             for w in self.tree_weights[level]:
#                 luts, ids = self.parametrization.get_luts_and_ids(w)
#                 level_ids.append(ids)
#                 level_luts.append(luts)
#             tree_ids.append(level_ids)
#             tree_luts.append(level_luts)
#         return tree_luts, tree_ids
    
#     def get_luts(self):
#         """Computes the most probable LUT for each neuron.

#         Returns:
#            List[List[torch.Tensor]]: Nested list of Boolean tensors (LUTs)
#         """
#         tree_luts = []
#         for level in range(self.tree_depth):
#             level_luts = []
#             for w in self.tree_weights[level]:
#                 luts = self.parametrization.get_luts(w)
#                 level_luts.append(luts)
#             tree_luts.append(level_luts)
#         return tree_luts
    
#     def get_regularization_loss(self, regularizer: str):
#         reg_loss = 0.0
#         for w in self.tree_weights:
#             reg_loss += get_regularization_loss(w, regularizer)
#         return reg_loss
    
#     def rescale_weights(self, method):
#         for w in self.tree_weights:
#             rescale_weights(w, method)

#     def _init_connections(self):
#         # Setup connections
#         if self.init_method == "random":
#             kernels = self._get_random_receptive_field_tensor()
#         # elif self.init_method == "random-unique":
#         #     kernels = self._get_random_unique_receptive_field_tensor()
#         else:
#             raise ValueError(f"Unknown connections type: {self.init_method}")
#         # Build tree indices
#         return self._get_indices_from_kernel_tensor(kernels)


#     def _get_random_receptive_field_tensor(self):
#         """
#         Random sampling (with replacement).

#         Returns:
#             coords: (lut_rank, num_kernels, sample_size, 3)
#         """

#         c = self.channels
#         g = self.channel_group_size
#         device = self.device

#         sample_size = self.lut_rank ** (self.tree_depth - 1)
#         total_inputs = self.lut_rank * sample_size

#         # ---------------------------
#         # Precompute spatial grid
#         # ---------------------------
#         rf_axes = [
#             torch.arange(0, dim, device=device)
#             for dim in self.receptive_field_size
#         ]

#         spatial_grid = torch.meshgrid(*rf_axes, indexing="ij")
#         spatial_positions = torch.stack(
#             [grid.flatten() for grid in spatial_grid], dim=1
#         )
#         num_spatial = spatial_positions.shape[0]

#         # ---------------------------
#         # Channel group setup
#         # ---------------------------
#         if g is None:
#             starts = None
#         else:
#             starts = torch.arange(0, c - g + 1, device=device)
#             num_groups = starts.numel()

#         coords_per_kernel = []

#         for k in range(self.num_kernels):

#             if g is None:
#                 c_rf = torch.arange(0, c, device=device)

#                 # full 3D position space
#                 grid = torch.meshgrid(*rf_axes, c_rf, indexing="ij")
#                 all_positions = torch.stack(
#                     [grid_i.flatten() for grid_i in grid], dim=1
#                 )
#                 num_positions = all_positions.shape[0]

#                 idx = torch.randint(
#                     0, num_positions,
#                     (sample_size, self.lut_rank),
#                     device=device,
#                 )

#                 coords_k = all_positions[idx]


#             coords_per_kernel.append(coords_k)

#         coords = torch.stack(coords_per_kernel, dim=0)
#         coords = coords.permute(2, 0, 1, 3)

#         return coords
    
#     def _apply_sliding_window_tensor(self, tensor):
#         """Apply sliding window offsets to receptive field tensor.

#         Args:
#             tensor: torch.Tensor of shape (lut_rank, num_kernels, sample_size, 3)
#                 where last dim is (h, w, c).

#         Returns:
#             out: torch.Tensor of shape (lut_rank, num_kernels, num_positions, sample_size, 3),
#                 with the sliding-window offsets applied.
#         """
#         #h, w = self.in_dim
#         #h_k, w_k = self.receptive_field_size

#         # Account for padding
#         padded = [in_dim + 2 * self.padding for in_dim in self.in_dim]
#         #h_padded = h + 2 * self.padding
#         #w_padded = w + 2 * self.padding

#         assert all(rfs <= p for rfs, p in zip(self.receptive_field_size, padded)), (
#             f"Receptive field size {self.receptive_field_size} must fit within input "
#             f"dimensions {padded} after padding."
#         )

#         # Sliding positions
#         starts = [torch.arange(0, p - rcf + 1, self.stride, device=self.device) 
#                   for p, rcf in zip(padded, self.receptive_field_size)]
#         #h_starts = torch.arange(0, padded[0] - self.receptive_field_size[0] + 1, self.stride, device=self.device)
#         #w_starts = torch.arange(0, padded[1] - self.receptive_field_size[1] + 1, self.stride, device=self.device)

#         # Meshgrid for all receptive-field start positions
#         grid = torch.meshgrid(*starts, indexing="ij")
#         offsets = [g.flatten() for g in grid]
#         num_positions = [o.numel() for o in offsets]

#         # tensor: (L, K, S, 3) → (K, L, S, 3)
#         pairs_all = tensor.permute(1, 0, 2, 3)
#         # K, L, S, _ = pairs_all.shape

#         # Split h, w, c coordinates: (K, L, S)
#         base = [pairs_all[..., i] for i in range(len(offsets))]
#         #h_base = pairs_all[..., 0]
#         #w_base = pairs_all[..., 1]
#         c_base = pairs_all[..., -1]

#         # Add sliding-window offsets (broadcasted) → (K, P, L, S)
#         idx = [b.unsqueeze(1) + o.view(1, num_positions[0], 1, 1) 
#                for b, o in zip(base, offsets)]
#         c_idx = c_base.unsqueeze(1).expand(-1, num_positions[0], -1, -1)

#         # Combine back into indices: (K, P, L, S, 3)
#         all_indices = torch.stack([*idx, c_idx], dim=-1)

#         # Reorder so first axis is L: (L, K, P, S, 3)
#         out = all_indices.permute(2, 0, 1, 3, 4)

#         return out

#     def _get_indices_from_kernel_tensor(self, tensor):
#         """Build index tensors for all tree levels."""
#         indices = [
#             self._apply_sliding_window_tensor(tensor)
#         ]
#         for level in range(1, self.tree_depth):
#             size = self.lut_rank ** (self.tree_depth - level)
#             base = torch.arange(size, device=self.device).view(-1, self.lut_rank).transpose(0, 1)
#             indices.append(base)
#         return indices


# class LogicConv2d(_LogicConvNd):
#     """2D convolutional layer with differentiable logic operations.

#     This layer implements a 2D convolution where each output location is
#     computed by evaluating a learned logic tree over a receptive field.
#     Instead of linear filters, it uses a binary tree of differentiable
#     logic operations (LUTs) applied to selected positions in the receptive
#     field, per kernel and per spatial location.
#     """
#     def __init__(
#         self,
#         in_dim: Union[_size_2_t, int],
#         channels: int = 1,
#         num_kernels: int = 16,
#         tree_depth: int = None,
#         receptive_field_size: Union[_size_2_t, int] = 2,
#         stride: int = 1,
#         padding: int = 0,
#         device: str = "cpu",
#         grad_factor: float = 1.0,
#         lut_rank: int = 2,
#         parametrization: str = "raw",
#         parametrization_kwargs: dict = None,
#         connections: str = "fixed",
#         connections_kwargs: dict = None,
#     ):
#         super().__init__(
#             in_dim=in_dim,
#             channels=channels,
#             num_kernels=num_kernels,
#             tree_depth=tree_depth,
#             receptive_field_size=receptive_field_size,
#             stride=stride,
#             padding=padding,
#             conv_dimension=2,
#             device=device,
#             grad_factor=grad_factor,
#             lut_rank=lut_rank,
#             parametrization=parametrization,
#             parametrization_kwargs=parametrization_kwargs,
#             connections=connections,
#             connections_kwargs=connections_kwargs,
#         )


