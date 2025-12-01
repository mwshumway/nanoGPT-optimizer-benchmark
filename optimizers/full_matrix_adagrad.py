# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Optional
from collections.abc import MutableMapping

import torch as torch
from torch import nn
from torch.optim import Optimizer


def _symmetric_matrix_power(
    mat: torch.Tensor,
    power: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute symmetric matrix power mat^power via eigendecomposition.

    Assumes mat is symmetric PSD. 
    """
    if mat.numel() == 0:
        return mat

    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals_clamped = eigvals.clamp(min=eps)
    powered = eigvals_clamped.pow(power)  # (d,)
    return (eigvecs * powered.unsqueeze(0)) @ eigvecs.t()


class FullMatrixAdaGrad(Optimizer):
    """
    Full-matrix AdaGrad optimizer.

    For each parameter vector w of dimension d, maintains a full d x d
    second-moment matrix:

        G_t = G_{t-1} + g_t g_t^T

    and updates:

        w_{t+1} = w_t - lr * (G_t + eps I)^{-1/2} g_t

    This is theoretically nice but O(d^2) memory and compute, so it's only
    practical for relatively small parameter tensors.

        params (`Iterable[torch.nn.Parameter]`):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (`float`, optional, default: 1e-2):
            Learning rate.
    Arguments:
        eps (`float`, optional, default: 1e-8):
            Numerical stability term added inside the matrix before inverting.
        precond_update_interval (`int`, optional, default: 1):
            How often (in steps) to recompute the matrix inverse square root.
            Setting > 1 can save some eigendecomp cost.
        max_precond_dim (`int`, optional, default: 4096):
            If a parameter has more than `max_precond_dim` elements, this
            optimizer will raise an error (to avoid exploding memory).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        eps: float = 1e-4,
        precond_update_interval: int = 1,
        max_precond_dim: int = 4096,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps} (should be > 0)")
        if precond_update_interval < 1:
            raise ValueError(
                f"precond_update_interval should be >= 1, got {precond_update_interval}"
            )
        if max_precond_dim <= 0:
            raise ValueError(
                f"max_precond_dim should be > 0, got {max_precond_dim}"
            )

        defaults = dict(
            lr=lr,
            eps=eps,
            precond_update_interval=precond_update_interval,
            max_precond_dim=max_precond_dim,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            precond_update_interval = group["precond_update_interval"]
            max_precond_dim = group["max_precond_dim"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "FullMatrixAdaGrad does not support sparse gradients"
                    )
                
            
                grad = p.grad.detach()
                g_flat = grad.view(-1)
                #print(g_flat.shape) debug used
                d = g_flat.numel()
                #print(d)
                if d > max_precond_dim:
                    raise RuntimeError(
                        f"Parameter with dimension {d} exceeds max_precond_dim={max_precond_dim} "
                        "for FullMatrixAdaGrad (would be too large O(d^2))."
                    )
                
                p_flat = p.data.view(-1)
                state = self.state[p]

                # Initialize state: allocate G and G_inv_sqrt on CPU explicitly
                if "step" not in state:
                    state["step"] = 0
                    # Keep G and G_inv_sqrt on CPU to avoid GPU OOM
                    state["G"] = torch.zeros(d, d, device="cpu", dtype=torch.float32)
                    state["G_inv_sqrt"] = torch.eye(d, device="cpu", dtype=torch.float32)

                state["step"] += 1
                step = state["step"]
                G = state["G"]  # CPU tensor
                G_inv_sqrt = state["G_inv_sqrt"]  # CPU tensor

                # Move gradient to CPU (float32) for accumulation / eigendecomp
                # Doing a single CPU<->GPU round-trip per parameter per step.
                g_flat_f32_cpu = g_flat.to(dtype=torch.float32, device="cpu")

                # Update CPU-side second moment
                # G <- G + g g^T (on CPU)
                G.add_(torch.outer(g_flat_f32_cpu, g_flat_f32_cpu))

                # Recompute inverse sqrt on CPU occasionally
                if step % precond_update_interval == 0:
                    mat = G + eps * torch.eye(d, device="cpu", dtype=G.dtype)
                    # compute inverse sqrt on CPU
                    G_inv_sqrt = _symmetric_matrix_power(mat, power=-0.5, eps=eps)
                    state["G_inv_sqrt"] = G_inv_sqrt

                # Precondition gradient on CPU
                precond_grad_f32_cpu = G_inv_sqrt @ g_flat_f32_cpu  # CPU

                # Move preconditioned gradient back to parameter device and original dtype
                precond_grad = precond_grad_f32_cpu.to(dtype=p_flat.dtype, device=p_flat.device)

                # Update parameter in-place on its device
                p_flat.add_(precond_grad, alpha=-lr)

        return loss
