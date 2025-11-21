# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Optional
from collections.abc import MutableMapping

import torch
from torch import nn
from torch.optim import Optimizer

class Adafactor(Optimizer):
    """
    Implements Adafactor algorithm
    Parameters:
         params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
            The parameters are grouped into two groups: 
                those with param_type = linear (for which we use the Adafactor update with no momentum and factorized step sizes),
                and those with param_type = regular (for which we use the Adam update).
        lr (`float`, *optional*, defaults to 0.001):
            The base learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            beta parameters (b1, b2) for momentum and step size EMAs.
        eps (`float`, *optional*, defaults to 1e-06):
            epsilon for numerical stability.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias
    """
    
    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            correct_bias: bool = True
    ):
        # implement the same basic checks as in Adam
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        defaults = {"lr": lr, "betas": betas, "eps": eps, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    # first step for this param, initialize the state
                    state["step"] = 0
                    if group.get("param_type") == "linear":
                        # for linear parameters, we maintain the factored step sizes
                        state["row_sums"] = torch.zeros((p.shape[0],), dtype=p.dtype, device=p.device)
                        state["col_sums"] = torch.zeros((p.shape[1],), dtype=p.dtype, device=p.device)
                    else:
                        # for regular parameters, we maintain the Adam step sizes
                        state["grads_ema"] = torch.zeros_like(p.data)
                        state["squared_grads_ema"] = torch.zeros_like(p.data)
                
                state["step"] += 1

                if group.get("param_type") == "linear":
                    self._step_adafactor(p, grad, state, group)
                else:
                    self._step_adam(p, grad, state, group)
            
        return loss
    
    def _step_adam(self, p, grad, state, group):
        beta1, beta2 = group["betas"]

        grads_ema = state["grads_ema"]
        grads_ema.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
        
        squared_grads_ema = state["squared_grads_ema"]
        squared_grads_ema.mul_(beta2).add_(grad**2, alpha=(1.0 - beta2))

        step_size = group["lr"]
        if group["correct_bias"]:
            bc1 = 1.0 - beta1 ** state["step"]
            bc2 = 1.0 - beta2 ** state["step"]
            step_size = step_size * math.sqrt(bc2) / bc1
        
        p.addcdiv_(grads_ema, squared_grads_ema.sqrt().add_(group["eps"]), value=-step_size)

    def _step_adafactor(self, p, grad, state, group):
        beta2 = group["betas"][1] # don't need beta1

        row_sums = state["row_sums"]
        col_sums = state["col_sums"]

        # update the row and column sums
        grad_squared = grad**2
        row_sums.mul_(beta2).add_(grad_squared.sum(dim=1), alpha=(1.0 - beta2))
        col_sums.mul_(beta2).add_(grad_squared.sum(dim=0), alpha=(1.0 - beta2))

        v_hat = torch.outer(row_sums, col_sums) / (row_sums.sum() + group["eps"])
        v_hat /= (1 - beta2 ** state["step"])  # bias correction

        p.addcdiv_(grad, v_hat.sqrt().add_(group["eps"]), value=-group["lr"])
