import os
import torch
import pandas as pd
import utils.cut_words as cw

from typing import Any, Dict
from torch.utils.data import Dataset


class DoubanCommentDataset(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        self.split = split
        self.csv_file_path = os.path.join(cfg.get("data_root"), 'DMSC.csv')
        self.split_ratio = cfg.get("split_ratio")
        self.data_max_num = cfg.get("data_max_num")

        if cfg.get("tokenizer") == "jieba":
            self.tokenizer = cw.JiebaTokenizer()
        elif cfg.get("tokenizer") == "SentencePiece":
            self.tokenizer = cw.SentencePieceTokenizer(vocab_size=cfg.get("vocab_size", 10000))

        self.data = self._load_data(cfg)
        
        if split == "train":
            cfg.update({
                'text_len': self.max_len,
                'vocab_size': len(self.vocab_to_index),
                'vocab_to_index': self.vocab_to_index,
                'index_to_vocab': self.index_to_vocab
            })

    def _load_data(self, cfg: Dict) -> pd.DataFrame:
        raw = pd.read_csv(self.csv_file_path, usecols=["Comment", "Star"])
        data = raw.iloc[:self.data_max_num].reset_index(drop=True)
        print(f"data use: {min(self.data_max_num, len(raw))} / {len(raw)}")

        split_idx = int(len(data) * self.split_ratio)
        data = data.iloc[:split_idx] if self.split == "train" else data.iloc[split_idx:]

        if self.split == "train":
            seqs, self.vocab_to_index, self.index_to_vocab, self.max_len = cw.texts_to_index_sequences(
                data["Comment"].tolist(),
                self.tokenizer,
                train=True
            )
        else:
            self.vocab_to_index = cfg["vocab_to_index"]
            self.max_len = cfg["text_len"]
            seqs = cw.build_eval_sequences(
                data["Comment"].tolist(),
                self.tokenizer,
                self.max_len
            )

        data["Comment"] = pd.Series(list(seqs), index=data.index)
        data.loc[:, "Star"] = (data["Star"] <= 2).astype(int)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        content = torch.tensor(self.data.iloc[idx]["Comment"], dtype=torch.long)
        label   = torch.tensor(self.data.iloc[idx]["Star"],  dtype=torch.long)
        return {'x': content, 'label': label}
    