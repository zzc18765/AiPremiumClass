import torch
import pandas as pd
import utils

from typing import Dict
from torch.utils.data import Dataset

import utils.cut_words as cw


class DoubanCommentDataset(Dataset):
    def __init__(self, split: str, cfg: Dict):
        self.split = split
        self.csv_file_path = cfg.get("data_root")
        self.split_ratio = cfg.get("split_ratio")
        self.data_max_num = cfg.get("data_max_num")

        self.data = self._load_data(cfg)
        
        if split == "train":
            cfg['text_len'] = self.max_len
            cfg['vocab_size'] = len(self.vocab_to_index)

    def _load_data(self, cfg: Dict) -> pd.DataFrame:
        raw = pd.read_csv(self.csv_file_path, usecols=["Comment", "Star"])
        data = raw.iloc[:self.data_max_num].reset_index(drop=True)
        print(f"data use: {min(self.data_max_num, len(raw))} / {len(raw)}")

        split_idx = int(len(data) * self.split_ratio)
        data = data.iloc[:split_idx] if self.split == "train" else data.iloc[split_idx:]

        seqs, self.vocab_to_index, self.index_to_vocab, self.max_len = utils.cut_words.texts_to_index_sequences(
            data["Comment"].tolist()
        )
        if self.split == "train":
            seqs, vocab_to_idx, idx_to_vocab, max_len = cw.texts_to_index_sequences(
                data["Comment"].tolist()
            )
            cfg.update({
                "vocab_to_index": vocab_to_idx,
                "index_to_vocab": idx_to_vocab,
                "vocab_size":     len(vocab_to_idx),
                "text_len":       max_len
            })
        else:
            vocab_to_idx = cfg["vocab_to_index"]
            max_len      = cfg["text_len"]
            seqs = cw.build_eval_sequences(
                data["Comment"].tolist(), vocab_to_idx, max_len
            )

        self.vocab_to_index = vocab_to_idx
        self.max_len        = max_len

        data["Comment"] = pd.Series(list(seqs), index=data.index)
        data.loc[:, "Star"] = (data["Star"] <= 2).astype(int)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        content = torch.tensor(self.data.iloc[idx]["Comment"], dtype=torch.long)
        label   = torch.tensor(self.data.iloc[idx]["Star"],  dtype=torch.long)
        return content, label
    