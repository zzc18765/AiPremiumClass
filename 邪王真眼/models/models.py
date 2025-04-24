import torch
import torch.nn as nn

from enum import Enum
from typing import Any, Dict


class ModelType(Enum):
    RNN_TEXT_CLASSIFIER = 'rnn_text_classifier'
    Logistic_Regression = 'logistic_regression'
    ResNet = 'resnet'
    RNN = 'rnn'
    CBOW = 'cbow'
    SkipGram = 'skip_gram'


def get_model(cfg: Dict[str, Any]):
    model_type = cfg.get('model')
    model = None

    if model_type == ModelType.RNN_TEXT_CLASSIFIER:
        from .rnn_classification import RNNTextClassifier
        model = RNNTextClassifier(cfg)
    elif model_type == ModelType.Logistic_Regression:
        from .logistic_regression import LogisticRegression
        model = LogisticRegression(cfg)
    elif model_type == ModelType.ResNet:
        from .resnet import ResNet
        model = ResNet(cfg)
    elif model_type == ModelType.RNN:
        from .rnn import RNN
        model = RNN(cfg)
    elif model_type == ModelType.CBOW:
        from .cbow import CBOW
        model = CBOW(cfg)
    elif model_type == ModelType.SkipGram:
        from .skip_gram import SkipGram
        model = SkipGram(cfg)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    return model
