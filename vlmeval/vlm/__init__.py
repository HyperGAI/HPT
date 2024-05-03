import torch
torch.set_grad_enabled(False)
torch.manual_seed(1234)

from .hpt import HPT
from .hpt1_5 import HPT1_5
