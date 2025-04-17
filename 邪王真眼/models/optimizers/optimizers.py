import torch.nn as nn
import torch.optim as optim

from enum import Enum
from typing import Dict


class OptimizerType(Enum):
    SGD = "SGD"


def get_optimizer(cfg: Dict[str, dict], model: nn.Module):
    def get_with_default(cfg, key, default_value):
        if key not in cfg:
            cfg[key] = default_value
            print(f"Warning: '{key}' not found in configuration. Using default value: {default_value}")
        return cfg.get(key, default_value)
    
    lr = get_with_default(cfg, 'lr', 0.01)
    momentum = get_with_default(cfg, 'momentum', 0.9)
    weight_decay = get_with_default(cfg, 'weight_decay', 1e-4)

    optimizer = cfg.get('optimizer')

    if optimizer == OptimizerType.SGD:
        return optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    return None
