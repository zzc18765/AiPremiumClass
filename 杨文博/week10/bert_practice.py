import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

df = pd.read_csv("jd_comment_w_label.csv")  # 确保路径正确
df = df.dropna()
df['label'] = df['label'].astype(int)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['comment'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

class JDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = JDDataset(train_texts, train_labels, tokenizer)
val_dataset = JDDataset(val_texts, val_labels, tokenizer)

def get_model(freeze_bert=False):
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-chinese",
        num_labels=len(set(df['label']))
    )
    if freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False
    return model

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

def train_model(freeze_bert):
    model = get_model(freeze_bert)
    tb_dir = f"./runs/freeze_{freeze_bert}"

    training_args = TrainingArguments(
        output_dir=f'./results/freeze_{freeze_bert}',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        logging_dir=tb_dir,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(f"./saved_model/freeze_{freeze_bert}")
    tokenizer.save_pretrained(f"./saved_model/freeze_{freeze_bert}")

train_model(freeze_bert=True)
train_model(freeze_bert=False)
