import torch.nn as nn

from enum import Enum
from typing import Dict


class LossFunctionType(Enum):
    CROSS_ENTROPY     = "cross_entropy"
    BCE_WITH_LOGITS   = "bce_with_logits"


def get_loss_function(cfg: Dict[str, dict]):
    loss_function = cfg.get('loss_function')
    if loss_function == LossFunctionType.CROSS_ENTROPY:
        return nn.CrossEntropyLoss(ignore_index=255)
    elif loss_function == LossFunctionType.BCE_WITH_LOGITS:
        return nn.BCEWithLogitsLoss()
    
    return None
