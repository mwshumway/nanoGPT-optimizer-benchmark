# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Optional
from collections.abc import MutableMapping

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version # type: ignore

# default values for Muon, reported from Keller Jordan's Muon post.
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5

class Adam(Optimizer):
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
        require_version("torch>=1.5.0")  # add_ with alpha
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

class AdamSN(Optimizer):
    """
    Implements Adam algorithm with Subset Norm (SN) variant.
    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
            The parameters are grouped into two groups: those with param_type = linear (for which we use row-wise step sizes),
            and those with param_type = regular (for which we use the per-coordinate step sizes).
        lr (`float`, *optional*, defaults to 0.001):
            The base learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            beta parameters (b1, b2) for momentum and step size EMAs.
        eps (`float`, *optional*, defaults to 1e-06):
            epsilon for numerical stability.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias
    Base step sizes:
        As in the Adam implementation above, we will use an additional base step size (lr) for the update, i.e.:
            x' = x - lr * m / (sqrt(v) + eps)
        For the subset norm variant, we will need to increase the provided lr by sqrt(size of the subset), as follows:
        for group in self.param_groups:
            for p in group["params"]:
                if group.get("param_type") == "linear":
                    lr = group["lr"]
                    # increase the step size by sqrt(row dimension)
                    lr = lr * math.sqrt(p.size(dim=1))
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
        require_version("torch>=1.5.0")  # add_ with alpha
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

                # second moment estimate v_t
                if group.get("param_type") == "linear":
                    second_moment_update = torch.sum(grad**2, dim=1, keepdim=True) # subset norm (row-wise) update
                else:
                    second_moment_update = grad**2 # regular adam update

                squared_grads_ema.mul_(beta2).add_(second_moment_update, alpha=(1.0 - beta2)) # v_t

                step_size = group["lr"]
                if group.get("param_type") == "linear":
                    # increase the step size by sqrt(row dimension)
                    step_size = step_size * math.sqrt(p.size(dim=1))

                # bias correction
                if group["correct_bias"]:
                    # incorporate the bias corrections into the step size
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # update the parameters
                p.addcdiv_(grads_ema, squared_grads_ema.sqrt().add_(group["eps"]), value=-step_size)

        return loss


    
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
        require_version("torch>=1.5.0")  # add_ with alpha
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


def _adjust_lr(
        lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size
) -> float:
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio

def newtonschulz(
        grad: torch.Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> torch.Tensor:
    
    if len(grad.shape) != 2:
        raise ValueError(f"Input gradient must be 2D, but got shape {grad.shape}")
    if len(ns_coefficients) != 3:
        raise ValueError(f"ns_coefficients must be a tuple of 3 floats, but got {ns_coefficients}")
    
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) >= grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    
    if grad.size(0) >= grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad

class Muon(Optimizer):
    """
    Implements Muon algorithm
    """

    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            weight_decay: float = 0.1,
            momentum: float = 0.95,
            nesterov: bool = False,
            ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
            eps: float = EPS,
            ns_steps: int = DEFAULT_NS_STEPS,
            adjust_lr_fn: Optional[str] = None,
            betas: Tuple[float, float] = (0.9, 0.999),
            correct_bias: bool = True,
    ):
        # Some basic checks
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate must be >= 0, but is {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Momentum must be >= 0, but is {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Weight decay must be >= 0, but is {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in [
            "original",
            "match_rms_adamw",
        ]:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )
  
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "betas": betas,
            "correct_bias": correct_bias
        }
        super().__init__(params, defaults)
        
        # for group in self.param_groups:
        #     for p in group["params"]:
        #         if p.ndim != 2:
        #             raise ValueError(f"Muon only supports 2D parameters, but got parameter with shape {p.shape}")
    
    def _init_group(
            self,
            group: MutableMapping,
            params_with_grad: list[torch.Tensor],
            grads: list[torch.Tensor],
            muon_momentum_bufs: list[torch.Tensor]
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            
            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
            muon_momentum_bufs.append(state["momentum_buffer"])


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    if group["param_type"] == "linear":
                        state["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    else:
                        # for regular parameters, we use Adam update
                        state["grads_ema"] = torch.zeros_like(p.data)
                        state["squared_grads_ema"] = torch.zeros_like(p.data)
                state["step"] += 1
            
                if group.get("param_type") == "linear":
                    self._step_muon(p, grad, state, group)
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
    
    def _step_muon(self, p, grad, state, group):
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        weight_decay = group["weight_decay"]
        ns_coefficients = group["ns_coefficients"]
        ns_steps = group["ns_steps"]
        eps = group["eps"]

        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(
            group["lr"], group["adjust_lr_fn"], p.shape
        )

        p.mul_(1 - adjusted_lr * weight_decay)
        p.add_(update, alpha=-adjusted_lr)


class FullMatrixAdaGrad(Optimizer):
    """
    Implements Full-Matrix AdaGrad algorithm
    """

    def __init__(self):
        raise NotImplementedError("Full-Matrix AdaGrad is not implemented yet.")


class Shampoo(Optimizer):
    """
    Implements Shampoo algorithm
    """

    def __init__(self):
        raise NotImplementedError("Shampoo is not implemented yet.")


class unScion(Optimizer):
    """
    Implements unScion algorithm
    """

    def __init__(self):
        raise NotImplementedError("unScion is not implemented yet.")
