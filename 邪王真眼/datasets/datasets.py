from enum import Enum
from typing import Any, Dict
from torch.utils.data import DataLoader


class DatasetType(Enum):
    DOUBAN_COMMENTS = 'douban_comments'
    CIFAR10 = "cifar10"
    SEM_EVAL = "sem_eval"
    SKIP_GRAM = "skip_gram"
    CBOW = "cbow"


def get_dataset(cfg: Dict[str, Any]):
    dataset_type = cfg.get('dataset')
    batch_size = cfg.get('batch_size')

    if dataset_type == DatasetType.DOUBAN_COMMENTS:
        from .douban_comments.generator import DoubanCommentDataset
        train_dataset = DoubanCommentDataset('train', cfg)
        val_dataset = DoubanCommentDataset('val', cfg)
    elif dataset_type == DatasetType.CIFAR10:
        from .cifar10.generator import CIFAR10Dataset
        train_dataset = CIFAR10Dataset('train', cfg)
        val_dataset = CIFAR10Dataset('val', cfg)
    elif dataset_type == DatasetType.SEM_EVAL:
        from .sem_eval.generator import SemEvalDataset
        train_dataset = SemEvalDataset('train', cfg)
        val_dataset = SemEvalDataset('val', cfg)
    elif dataset_type == DatasetType.SKIP_GRAM:
        from .word2vec.generator_skip_gram import Skipgram
        train_dataset = Skipgram('train', cfg)
        val_dataset = Skipgram('val', cfg)
    elif dataset_type == DatasetType.CBOW:
        from .word2vec.generator_cbow import CBOW
        train_dataset = CBOW('train', cfg)
        val_dataset = CBOW('val', cfg)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
