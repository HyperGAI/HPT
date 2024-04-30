import torch
torch.set_grad_enabled(False)
torch.manual_seed(1234)

from .hpt import HPT
from .fuyu import Fuyu8B
