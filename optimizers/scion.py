"""
Adapted from
https://github.com/LIONS-EPFL/scion/blob/main/scion.py 
"""

# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Optional
from collections.abc import MutableMapping

import torch
from torch import nn
from torch.optim import Optimizer
import math

from abc import ABCMeta, abstractmethod

class Norm(object, metaclass=ABCMeta):
    @abstractmethod
    def lmo(self, g):
        raise NotImplementedError

    @abstractmethod
    def init(self, w):
        raise NotImplementedError

class ColNorm(Norm):
    """
    Column-wise normalization

    Args:
        normalized (bool, optional); whether to normalize input tensor (use only for non-input layers)
        transpose (bool, optional): whether to transpose input before normalization (use only for embedding layers)
    """
    def __init__(self, normalized: bool = False, transpose: bool = False):
        self.normalized = normalized
        self.transpose = transpose
    
    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1)
        rms_vals = 1 / math.sqrt(g.size(0)) * torch.sqrt(torch.sum(g ** 2, dim=0, keepdim=True) + eps)
        if self.normalized:
            rms_vals *= g.size(1)
        g = g / (rms_vals + eps)
        if self.transpose:
            g = g.transpose(0, 1)
        return g

    def init(self, w):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=0, keepdim=True)
        w.data *= math.sqrt(w.size(0))
        if self.normalized:
            w.data /= w.size(1)
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        return w
    
class Sign(Norm):
    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, g):
        d = g.shape
        if len(d) != 2:
            d_in = d[-1]
        else:
            d_out, d_in = d
        if self.normalized:
            return (1/d_in)*torch.sign(g)    
        else:
            return torch.sign(g)

    def init(self, w):
        if self.zero_init:
            torch.nn.init.zeros_(w)
        else:
            # Generate -1/fan_in or 1/fan_in uniformly at random
            d = w.shape
            if len(d) != 2:
                d_in = d[-1]
            else:
                d_out, d_in = d
            w.data = (torch.randint(0, 2, w.shape, dtype=w.dtype, device=w.device) * 2 - 1)
            if self.normalized:
                w.data *= (1/d_in)
        return w


class Spectral(Norm):
    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.normalized = normalized
        self.steps = steps
    
    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        d_out, d_in = g.shape

        if self.normalized:
            scale = (d_out / d_in) ** 0.5
        else:
            scale = d_out ** 0.5
        if self.max:
            scale = max(1, scale)
        g *= scale

        return g
    
    def init(self, w):
        assert len(w.shape) == 2, f"Spectral init only supports 2D tensors, got {w.shape}"

        w_fp = w.data.float() # float32
        torch.nn.init.orthogonal_(w_fp)
        d_out, d_in = w_fp.shape
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        w_fp.mul_(scale)
    
        w.data = w_fp.to(dtype=w.data.dtype)
        return w        


def zeropower_via_newtonschulz5(G, steps=5):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X


norm_dict = {
    'ColNorm': ColNorm,
    'Spectral': Spectral,
    'Sign': Sign,
}

class Scion(Optimizer):
    """
    Implements Scion algorithm
    """

    def __init__(self, params, lr=1e-3, momentum=1.0, norm: str='Auto', norm_kwargs: dict=None, scale=1.0, unconstrained=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(lr=lr, momentum=momentum, scale=scale, unconstrained=unconstrained, norm=norm, norm_kwargs=norm_kwargs)
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if momentum != 1:
                    if 'momentum_buffer' not in state.keys():
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(1-momentum).add_(g, alpha=momentum)
                    g = buf

                update = scale * norm_backend.lmo(g)
                if not unconstrained:
                    p.data.mul_(1-lr)
                p.data.add_(update, alpha=-lr)

    def init(self):
        for group in self.param_groups:
            if group['norm'] == 'Auto':
                print(group)
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            init_func = norm_backend.init
            scale = group['scale']
            for p in group['params']:
                init_func(p)
                p.data *= scale