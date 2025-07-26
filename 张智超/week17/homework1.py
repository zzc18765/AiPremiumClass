import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.adamw

def get_batch(split):
    list = train_data if split == 'train' else val_data
    nl = torch.randint(0, len(list) - block_size -1, (batch_size,))
    xb = torch.stack([list[n: n+block_size] for n in nl])
    yb = torch.stack([list[n+1: n+1+block_size] for n in nl])
    return xb.to(device), yb.to(device)

class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_x):
        # input_x.shape = (batch_size, token_size, n_embd)
        B, T, C = input_x.shape
        q = self.query(input_x) # B, T, head_size
        k = self.key(input_x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        tril = torch.tril(torch.ones((T, T), device=device))
        wei = wei.masked_fill(tril == 0, float('-inf')) # (B, T, T)
        wei = wei.softmax(dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(input_x)
        out = wei @ v # B, T, head_size
        return out

class MultiHeadAttention(nn.Module):
    """"多头注意力"""
    def __init__(self, n_embd, head_num, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(head_num)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_x):
        out = torch.cat([h(input_x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, input_x):
        return self.net(input_x)

class Block(nn.Module):
    """自注意力模块"""
    def __init__(self, n_embd, head_num, dropout):
        super().__init__()
        head_size = n_embd//head_num
        self.sa = MultiHeadAttention(n_embd, head_num, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, input_x):
        x = input_x + self.sa(self.ln1(input_x)) # 残差连接
        x = x + self.ffwd(self.ln2(x)) # 残差连接
        return x

class BingramLanguageModel(nn.Module):
    def __init__(self, block_size, vocab_size, n_embd, head_num, dropout, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # 位置编码
        self.blocks = nn.Sequential(
            *[Block(n_embd, head_num, dropout) for _ in range(n_layer)],
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # B, T, n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, n_embd
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # B, T, vocab_size

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_tokens):
        # 初始idx.shape=(1,1)
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond) # (1, token_size, emb_dim)
            logits = logits[:, -1, :] # 取最后一个token的logits
            probs = F.softmax(logits, dim=-1) # 使用softmax转换为概率值，因为torch.multinomial需要概率值
            # 进行随机采样
            idx_next = torch.multinomial(probs, num_samples=1) # (1,1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == '__main__':
    with open("./files/three_body_part.txt", encoding="utf-8") as f:
        text = f.read()
    charts = sorted(list(set(text)))
    # 词典大小
    vocab_size = len(charts)
    ctoi = { c: i for i, c in enumerate(charts)}
    itoc = { i: c for i, c in enumerate(charts)}
    # 编码器：句子-》index
    encoder = lambda s: [ctoi[c] for c in s]
    # 解码器：index -》句子
    decoder = lambda l: ''.join([itoc[i] for i in l])

    # 数据集
    data = torch.tensor(encoder(text), dtype=torch.long)
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]

    # 训练语料
    block_size = 32
    batch_size = 32
    epoches = 5000
    lr = 1e-3
    n_embd = 32
    head_num = 4
    dropout = 0.1
    n_layer = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 训练
    model = BingramLanguageModel(block_size, vocab_size, n_embd, head_num, dropout, n_layer)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epoches):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 500 == 0: print(loss.item())

    # 生成
    token_idx = torch.zeros((1,1), dtype=torch.long, device=device)
    result = model.generate(token_idx, 500)
    content = decoder(result[0].tolist())
    print(content)

    
    

