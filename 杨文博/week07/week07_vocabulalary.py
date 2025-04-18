import pickle
from week07_classify import Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split


class Vocabulary:
    def __init__(self):
        self.vocabulary = set()
        self.word2index = {}
        self.index2word = {}

    def load_build(self, path):
        with open(path, 'rb') as f:
            comments_data = pickle.load(f)
        print("示例数据:", comments_data[0])

        for comment in comments_data:
            self.vocabulary.update(comment[0].split())
        print("词汇表大小:", len(self.vocabulary))

        self.word2index = {word: idx + 2 for idx, word in enumerate(self.vocabulary)}
        self.word2index['<PAD>'] = 0
        self.word2index['<UNK>'] = 1
        self.index2word = {idx: word for word, idx in self.word2index.items()}


class DataPrepare:
    def __init__(self, word2index):
        self.word2index = word2index
        self.index_comments = []
        self.labels = []

    def load_data(self, path):
        with open(path, 'rb') as f:
            comments_data = pickle.load(f)

        for comment in comments_data:
            idx_seq = [self.word2index.get(word, self.word2index['<UNK>'])
                       for word in comment[0].split()]
            self.index_comments.append(idx_seq)
            self.labels.append(comment[1])

        print(f"加载了 {len(self.index_comments)} 条评论数据")


def pad_sequences(sequences, max_len=None, pad_value=0):
    """填充序列到相同长度"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded = np.zeros((len(sequences), max_len))
    lengths = []
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        lengths.append(seq_len)
        padded[i, :seq_len] = seq[:max_len]
    return padded, lengths


class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MyEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = MyEmbedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, lengths=None):
        embedded = self.embedding(input_ids)

        if lengths is not None:
            # 打包序列以处理变长输入
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False)

        _, hidden = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels, lengths in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, lengths in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct / total


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.lengths = [len(seq) for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]


def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    sequences_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq) for seq in sequences],
        batch_first=True,
        padding_value=0
    )
    return sequences_padded, torch.tensor(labels), torch.tensor(lengths)


if __name__ == '__main__':
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 构建词汇表
    vocabulary = Vocabulary()
    vocabulary.load_build(Config.JIEBA_DATA_PATH)

    # 2. 准备数据
    data_prepare = DataPrepare(vocabulary.word2index)
    data_prepare.load_data(Config.JIEBA_DATA_PATH)

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        data_prepare.index_comments,
        data_prepare.labels,
        test_size=0.2,
        random_state=42
    )

    # 4. 创建数据集和数据加载器
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn
    )

    # 5. 初始化模型
    vocab_size = len(vocabulary.word2index)
    num_classes = len(set(data_prepare.labels))

    model = TextClassifier(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_classes=num_classes
    ).to(device)

    # 6. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 7. 训练循环
    print("开始训练...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}:")
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2%}")
        print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2%}")
        print("-" * 50)

    # 8. 保存模型
    if Config.SAVE_MODEL:
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        print(f"模型已保存到 {Config.MODEL_SAVE_PATH}")