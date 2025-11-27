# copy dependencies from transformers/optimization.py
import math
from typing import Callable, Iterable, Optional
from torch.optim import Optimizer
import torch


def _matrix_power(mat, power, eps=1e-12):
    """Eigen-decomposition based matrix power."""
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp(min=eps)
    return (eigvecs * eigvals.pow(power).unsqueeze(0)) @ eigvecs.T


class Shampoo(Optimizer):
    """
    CPU Shampoo:
      - 2D parameters → full Shampoo update
      - 1D parameters → AdaGrad fallback
    """

    def __init__(self, params: Iterable, lr=1e-1, eps=1e-4):
        if lr < 0.0:
            raise ValueError("lr must be >= 0")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")

        defaults = {"lr": lr, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        Returns:
            loss (float or None): same behavior as Adam
        """
        loss = None
        if closure is not None:
            # closure runs forward/backward
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.detach().to(torch.float32)
                state = self.state[p]

  
                # 2D case: Shampoo
                if g.dim() == 2:
                    if "L" not in state:
                        m, n = g.shape
                        state["L"] = eps * torch.eye(m, dtype=torch.float32, device=g.device)
                        state["R"] = eps * torch.eye(n, dtype=torch.float32, device=g.device)

                    L = state["L"]
                    R = state["R"]

                    # accumulate
                    L.add_(g @ g.T)
                    R.add_(g.T @ g)

                    # inverse 4th-root
                    L_inv = _matrix_power(L, -0.25, eps)
                    R_inv = _matrix_power(R, -0.25, eps)

                    # precondition
                    g_pre = L_inv @ g @ R_inv

                    # update parameter
                    p.add_(g_pre.to(p.dtype), alpha=-lr)
                    continue
                
                # 1D case: AdaGrad
                elif g.dim() == 1:
                    if "sum_sq" not in state:
                        state["sum_sq"] = torch.zeros_like(g, dtype=torch.float32, device=g.device)

                    state["sum_sq"].add_(g * g)
                    denom = state["sum_sq"].sqrt() + eps


                    update = (-lr * g / denom).to(p.dtype)
                    p.add_(update)
                    continue

                else:
                    raise RuntimeError(
                        f"Shampoo only supports 1D or 2D parameters, but got shape={tuple(g.shape)}"
                    )

        return loss
