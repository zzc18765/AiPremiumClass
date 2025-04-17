import torch
import torch.nn as nn

from enum import Enum
from typing import Dict

from .rnn_classification import RNNTextClassifier


class ModelType(Enum):
    RNN_TEXT_CLASSIFIER = 'rnn_text_classifier'


def get_model(cfg: Dict[str, dict]):
    model_type = cfg.get('model')
    model = None

    if model_type == ModelType.RNN_TEXT_CLASSIFIER:
        model = RNNTextClassifier(cfg)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    return model
