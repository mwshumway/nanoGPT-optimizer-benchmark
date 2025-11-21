# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Optional
from collections.abc import MutableMapping

import torch
from torch import nn
from torch.optim import Optimizer


class Shampoo(Optimizer):
    """
    Implements Shampoo algorithm
    """

    def __init__(self):
        raise NotImplementedError("Shampoo is not implemented yet.")
    