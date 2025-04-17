from torch.utils.data import DataLoader

from enum import Enum
from typing import Dict

from .douban_comments.generator import DoubanCommentDataset


class DatasetType(Enum):
    DOUBAN_COMMENTS = 'douban_comments'


def get_dataset(cfg: Dict[str, dict]):
    dataset_type = cfg.get('dataset')
    batch_size = cfg.get('batch_size')

    if dataset_type == DatasetType.DOUBAN_COMMENTS:
        train_dataset = DoubanCommentDataset('train', cfg)
        val_dataset = DoubanCommentDataset('val', cfg)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader
