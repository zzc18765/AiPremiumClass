import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Config:
    max_seq_len = 64
    vocab_size = 3000
    embedding_dim = 256
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].detach()
        return x


class PoetryTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(embedding_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=2 * embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids, src_key_padding_mask=None):
        x = self.token_emb(input_ids)
        x = x * math.sqrt(self.token_emb.embedding_dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc(x)
        return logits

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )
