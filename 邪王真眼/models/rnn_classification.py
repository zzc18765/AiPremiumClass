import torch.nn as nn

from typing import Dict


class RNNTextClassifier(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.num_classes = cfg["num_classes"]

        self.embedding = nn.Embedding(
            num_embeddings=cfg["vocab_size"],
            embedding_dim=cfg.get("embed_dim", 300),
            padding_idx=1
        )

        self.rnn = nn.LSTM(
            input_size=cfg.get("embed_dim", 300),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            batch_first=True,
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
            bidirectional=False
        )

        self.fc = nn.Linear(cfg["hidden_size"], self.num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        
        _, (h_n, _) = self.rnn(emb)
        feat = h_n[-1]

        logits = self.fc(feat)
        return {'out': logits}
