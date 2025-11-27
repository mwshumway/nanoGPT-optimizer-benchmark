# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Optional
from collections.abc import MutableMapping

import torch
from torch import nn
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Implements Adam algorithm
    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The base learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            beta parameters (b1, b2) for first moment (m) and second moment (v) EMAs.
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
            correct_bias: bool = True,
    ):
        # some basic checks
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
                grad = p.grad # the parameter's gradient
                state = self.state[p] # the parameter's state

                if "step" not in state:
                    # this is the first step
                    # initialize the step count
                    state["step"] = 0
                    # initialize the momentum and stepsizes m, v
                    state["grads_ema"] = torch.zeros_like(p.data)
                    state["squared_grads_ema"] = torch.zeros_like(p.data)
            
                state["step"] += 1

                # update the momentum and stepsizes m, v
                beta1, beta2 = group["betas"]
                grads_ema = state["grads_ema"]
                grads_ema.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                squared_grads_ema = state["squared_grads_ema"]
                squared_grads_ema.mul_(beta2).add_(grad**2, alpha=(1.0 - beta2))

                # bias correction
                step_size = group["lr"]
                if group["correct_bias"]:
                    # incorporate the bias corrections into the step size
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # update the parameters
                p.addcdiv_(grads_ema, squared_grads_ema.sqrt().add_(group["eps"]), value=-step_size)

        return loss
    