"""Interpolating Neural Network (INN) for PyTorch.

A pure-PyTorch re-implementation of the INN architecture described in:

    Park et al., "Unifying machine learning and interpolation theory via
    interpolating neural networks", Nature Communications (2025).
    Reference code: https://github.com/hachanook/pyinn

The INN combines **1-D linear interpolation** with **CP (canonical
polyadic) tensor decomposition** to learn a function from data:

    f(x₁, …, x_D) ≈ Σₘ Πd φ_d^m(x_d)

where each φ_d^m is a piecewise-linear function whose *nodal values* are
the trainable parameters.  The result is a lightweight, fully
differentiable model that can be trained with standard optimisers.

Typical use::

    model = INNInterpolator(input_dim=2, n_modes=20, n_segments=20)
    model.fit(x_train, y_train, epochs=1000, lr=1e-3)

    y_pred = model.predict(x_test)
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ======================================================================
#  Core INN module
# ======================================================================

class INNInterpolator(nn.Module):
    """Interpolating Neural Network using CP tensor decomposition.

    **Architecture (per forward pass)**

    1. For each dimension *d* of the input, locate the grid cell
       that contains ``x_d`` and compute the piecewise-linear
       interpolation from learnable nodal values.
    2. Take the element-wise product across dimensions (CP product).
    3. Sum across the *M* modes to produce the output.

    Parameters
    ----------
    input_dim : int
        Number of input dimensions (D).
    output_dim : int
        Number of output dimensions (default 1).
    n_modes : int
        Number of CP modes (M).  More modes → more capacity.
    n_segments : int
        Number of grid segments per dimension.  The grid has
        ``n_segments + 1`` nodes.
    x_ranges : list of (float, float), optional
        Per-dimension ``(min, max)`` ranges.  If *None* they are set
        to ``(0, 1)`` for every dimension (call :meth:`fit` or
        :meth:`set_ranges` before predicting with raw data).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        n_modes: int = 20,
        n_segments: int = 20,
        x_ranges: Optional[list] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.n_segments = n_segments
        self.n_nodes = n_segments + 1

        # Learnable nodal values: (M, D, output_dim, n_nodes)
        self.params = nn.Parameter(
            torch.randn(n_modes, input_dim, output_dim, self.n_nodes) * 0.1
        )

        # Grid ranges per dimension
        if x_ranges is not None:
            mins = torch.tensor([r[0] for r in x_ranges], dtype=torch.float32)
            maxs = torch.tensor([r[1] for r in x_ranges], dtype=torch.float32)
        else:
            mins = torch.zeros(input_dim)
            maxs = torch.ones(input_dim)
        self.register_buffer("grid_min", mins)
        self.register_buffer("grid_max", maxs)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def set_ranges(self, x_ranges: list):
        """Set per-dimension ``(min, max)`` ranges after construction."""
        self.grid_min = torch.tensor(
            [r[0] for r in x_ranges],
            dtype=torch.float32,
            device=self.params.device,
        )
        self.grid_max = torch.tensor(
            [r[1] for r in x_ranges],
            dtype=torch.float32,
            device=self.params.device,
        )

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return self.n_modes * self.input_dim * self.output_dim * self.n_nodes

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the INN at query points.

        Parameters
        ----------
        x : Tensor of shape ``(batch, input_dim)``

        Returns
        -------
        Tensor of shape ``(batch, output_dim)``
        """
        batch = x.shape[0]

        # Normalise to [0, n_segments] per dimension
        # x: (batch, D), grid_min/max: (D,)
        x_norm = (x - self.grid_min) / (self.grid_max - self.grid_min) * self.n_segments

        # Clamp to valid range
        x_norm = x_norm.clamp(0.0, self.n_segments - 1e-6)

        # Integer cell index and fractional part
        idx = x_norm.long()                       # (batch, D)
        idx = idx.clamp(max=self.n_segments - 1)
        frac = x_norm - idx.float()               # (batch, D)

        # --- 1-D linear interpolation for every (mode, dim, var) --------
        # params shape: (M, D, V, n_nodes)
        # We need:  val_left  = params[:, d, :, idx[b, d]]  for each b, d
        #           val_right = params[:, d, :, idx[b, d]+1]
        # Result:   phi[b, m, d, v] = val_left * (1-f) + val_right * f

        # Gather indices: expand idx to (batch, M, D, V)
        M, D, V, _ = self.params.shape
        idx_exp = idx[:, None, :, None].expand(batch, M, D, V)  # (B, M, D, V)

        # params expanded: (1, M, D, V, n_nodes) → gather along last dim
        params_exp = self.params.unsqueeze(0).expand(batch, -1, -1, -1, -1)
        val_left = params_exp.gather(-1, idx_exp.unsqueeze(-1)).squeeze(-1)
        val_right = params_exp.gather(-1, (idx_exp + 1).unsqueeze(-1)).squeeze(-1)

        frac_exp = frac[:, None, :, None].expand_as(val_left)
        phi = val_left * (1.0 - frac_exp) + val_right * frac_exp  # (B, M, D, V)

        # --- CP product across dimensions & sum over modes --------------
        # product over D:  (B, M, V) = prod_d phi[b, m, d, v]
        cp = phi.prod(dim=2)      # (B, M, V)
        out = cp.sum(dim=1)       # (B, V)
        return out

    # ------------------------------------------------------------------
    #  Convenience training wrapper
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        epochs: int = 1000,
        lr: float = 1e-3,
        batch_size: int = 256,
        validation_split: float = 0.1,
        patience: int = 50,
        verbose: bool = True,
    ) -> dict:
        """Train the INN on data.

        Parameters
        ----------
        x_train : (N, D) tensor of input locations.
        y_train : (N,) or (N, V) tensor of target values.
        epochs, lr, batch_size : standard training hyper-parameters.
        validation_split : fraction of data used for early-stopping.
        patience : early-stopping patience (in validation checks).
        verbose : print progress every 10 % of epochs.

        Returns
        -------
        dict with ``train_losses``, ``val_losses``, ``train_time``.
        """
        device = self.params.device
        if y_train.dim() == 1:
            y_train = y_train.unsqueeze(-1)
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        # Auto-set ranges from data
        with torch.no_grad():
            data_min = x_train.min(dim=0).values
            data_max = x_train.max(dim=0).values
            margin = (data_max - data_min) * 0.01
            margin = margin.clamp(min=1e-8)
            self.grid_min.copy_(data_min - margin)
            self.grid_max.copy_(data_max + margin)

        # Split
        n = x_train.shape[0]
        n_val = max(1, int(n * validation_split))
        perm = torch.randperm(n, device=device)
        idx_val = perm[:n_val]
        idx_train = perm[n_val:]

        x_tr, y_tr = x_train[idx_train], y_train[idx_train]
        x_va, y_va = x_train[idx_val], y_train[idx_val]
        n_tr = x_tr.shape[0]

        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        best_val = float("inf")
        patience_ctr = 0
        train_losses, val_losses = [], []

        log_every = max(1, epochs // 10)

        t0 = time.perf_counter()
        for epoch in range(epochs):
            self.train()
            perm_tr = torch.randperm(n_tr, device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_tr, batch_size):
                end = min(start + batch_size, n_tr)
                bi = perm_tr[start:end]
                pred = self(x_tr[bi])
                loss = (pred - y_tr[bi]).pow(2).mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * (end - start)
                n_batches += 1
            epoch_loss /= n_tr
            train_losses.append(epoch_loss)

            # Validation
            self.eval()
            with torch.no_grad():
                val_pred = self(x_va)
                val_loss = (val_pred - y_va).pow(2).mean().item()
            val_losses.append(val_loss)

            if verbose and (epoch + 1) % log_every == 0:
                print(
                    f"  Epoch {epoch+1:>5d}/{epochs}  "
                    f"train_rmse={math.sqrt(epoch_loss):.4e}  "
                    f"val_rmse={math.sqrt(val_loss):.4e}"
                )

            if val_loss < best_val:
                best_val = val_loss
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break

        train_time = time.perf_counter() - t0

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_time": train_time,
        }

    # ------------------------------------------------------------------
    #  Prediction helper (consistent with benchmark API)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the trained model (no gradients).

        Parameters
        ----------
        x : (N, D) tensor.

        Returns
        -------
        (N,) or (N, V) tensor.
        """
        self.eval()
        out = self(x)
        if out.shape[-1] == 1:
            return out.squeeze(-1)
        return out
