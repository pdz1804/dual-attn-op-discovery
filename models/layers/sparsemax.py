# -*- coding: utf-8 -*-
"""
Sparsemax activation function.

PyTorch implementation of Sparsemax from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
"""

from __future__ import division
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sparsemax(nn.Module):
    """Sparsemax function with debug printing."""

    def __init__(self, dim=None, device=device, debug=False):
        """
        Initialize Sparsemax.

        Args:
            dim (int, optional): Dimension along which to apply sparsemax. Default: last dimension.
            device (torch.device): Device to use (e.g., 'cuda' or 'cpu').
        """
        super(Sparsemax, self).__init__()
        self.device = device
        self.dim = -1 if dim is None else dim
        self.debug = debug

    def forward(self, input_tensor):
        """
        Forward pass of sparsemax.

        Args:
            input_tensor (Tensor): Input tensor of shape (batch_size, num_logits) or higher-rank.

        Returns:
            Tensor: Sparsemax output with some values exactly zero.
        """
        # Move input to correct device
        input_tensor = input_tensor.to(self.device)

        # Debug: input shape and content
        if self.debug:
            print(f"Input tensor shape:\n {input_tensor.shape}")
            print(f"Input tensor:\n {input_tensor}")

        # Move the target dimension to the front to make processing easier
        input_tensor = input_tensor.transpose(0, self.dim)

        # Save the original shape to reshape back later
        original_size = input_tensor.size()

        # Flatten all dimensions except the first (to apply sparsemax row-wise)
        input_tensor = input_tensor.reshape(input_tensor.size(0), -1)
        input_tensor = input_tensor.transpose(0, 1)  # shape: (batch, num_logits)
        dim = 1  # apply sparsemax across logits

        num_logits = input_tensor.size(dim)

        # Subtract max per row for numerical stability (softmax trick)
        input_tensor = input_tensor - torch.max(input_tensor, dim=dim, keepdim=True)[0].expand_as(input_tensor)

        # Sort each row in descending order
        zs = torch.sort(input=input_tensor, dim=dim, descending=True)[0]

        # Create [1, 2, ..., num_logits] per row
        range = torch.arange(1, num_logits + 1, device=self.device, dtype=input_tensor.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Compute condition: 1 + k * z_k > sum_{j=1}^k z_j
        bound = 1 + range * zs
        cumsum_zs = torch.cumsum(zs, dim)
        is_gt = (bound > cumsum_zs).type(input_tensor.dtype)

        # Find max valid k per row
        k = torch.max(is_gt * range, dim=dim, keepdim=True)[0]  # k is scalar per row

        # Compute threshold tau per row
        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim=dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input_tensor)

        # Final sparsemax output: max(0, z_i - tau)
        self.output = torch.max(torch.zeros_like(input_tensor), input_tensor - taus)

        # Debug output for each row
        if self.debug:
            print(f"\n======== DEBUG INFO ========")
            print("Shifted input (z - max):\n", input_tensor)
            print("-----------")
            print("Sorted zs:\n", zs)
            print("-----------")
            print("Cumulative sum of zs:\n", cumsum_zs)
            print("-----------")
            print("is_gt mask (1 + k*z_k > sum):\n", is_gt)
            print("-----------")
            print("k (active set size):\n", k)
            print("-----------")
            print("tau (threshold):\n", taus)
            print("-----------")
            print("Sparsemax output:\n", self.output)
            print("============================\n")

        # Reshape back to original tensor shape
        output = self.output.transpose(0, 1).reshape(original_size)
        output = output.transpose(0, self.dim)  # restore original dim order

        return output

# Example usage:
# # Define input tensor: batch of 3 rows
# x = torch.tensor([
#     [3.0, 1.0, 0.0, -1.0, -2.0],     # One large value → sparsemax will give 1-hot
#     [1.0, 2.0, 3.0, 4.0, 5.0],       # Smoothly increasing → 2 values may be active
#     [2.0, 2.0, 2.0, 2.0, 2.0],       # All equal → uniform distribution
#     [-1.0, -2.0, -3.0, -4.0, -5.0],  # All negative → only one likely to be non-zero
# ], dtype=torch.float32)

# # Instantiate sparsemax
# sparsemax = Sparsemax(dim=1, device=device, debug=True)

# # Run forward pass
# output = sparsemax(x)
# print("\nFinal Output:")
# print(output)

