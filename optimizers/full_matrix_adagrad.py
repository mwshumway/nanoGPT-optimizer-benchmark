# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Optional
from collections.abc import MutableMapping

import torch
from torch import nn
from torch.optim import Optimizer


class FullMatrixAdaGrad(Optimizer):
    """
    Implements Full-Matrix AdaGrad algorithm
    """

    def __init__(self):
        raise NotImplementedError("Full-Matrix AdaGrad is not implemented yet.")