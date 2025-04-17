from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.RNN_TEXT_CLASSIFIER,
    num_classes = 2,
    hidden_size = 300,
    embed_dim = 300,
    num_layers = 2,
    dropout = 0.3,

    dataset = DatasetType.DOUBAN_COMMENTS,
    split_ratio = 0.7,
    batch_size = 128,
    data_max_num = 10000,
    data_root = './邪王真眼/datasets/douban_comments/DMSC.csv',

    optimizer = OptimizerType.SGD,
    lr = 0.01,
    momentum = 0.9,
    weight_decay = 0.0001,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 50,
    result_path = './邪王真眼/week07/result'
)
