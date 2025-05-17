import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=5000):
        super().__init__()

        # 构建位置编码矩阵 [max_seq_len, embedding_dim]
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        # 偶数位：sin(position / (10000^(2i/dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位：cos(position / (10000^(2i/dim)))
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度：[1, max_seq_len, embedding_dim]
        pe = pe.unsqueeze(0)

        # 注册为 buffer，不作为参数参与训练，但保存到模型中
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        # 加 positional encoding（不参与训练）
        x = x + self.pe[:, :seq_len, :].detach()
        return x


class PoetryTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout, max_seq_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(embedding_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=embedding_dim * 2, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, dim_feedforward=embedding_dim * 2, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        self.output_proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src_ids, tgt_ids, tgt_mask=None, src_key_padding_mask=None):
        # src_ids: [batch_size, src_len]
        # tgt_ids: [batch_size, tgt_len]

        # Embedding + Positional Encoding
        src = self.embedding(src_ids) * math.sqrt(self.embedding.embedding_dim)  # [batch_size, seq_len, embedding_dim]
        src = self.pos_encoder(src)
        src = self.dropout(src)
        src = src.transpose(0, 1)  # [batch_size, seq_len, embedding_dim]

        tgt = self.embedding(tgt_ids) * math.sqrt(self.embedding.embedding_dim)  # [batch_size, seq_le, embedding_dim]
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)
        tgt = tgt.transpose(0, 1)  # [batch_size, seq_len, embedding_dim]

        # Transformer encoder-decoder
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)  # [seq_len, batch_size, embedding_dim]
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # [seq_len, batch_size, embedding_dim]

        output = output.transpose(0, 1)  # [batch_size, seq_len, embedding_dim]
        logits = self.output_proj(output)  # [batch_size, seq_len, vocab_size]

        return logits