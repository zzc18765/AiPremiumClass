import torch
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import evaluate
from datasets import load_dataset

# 1. 初始化模型和tokenizer
model_name = 'google-bert/bert-base-chinese'

# MSRA NER 数据集的标签（根据数据集实际情况调整）
tags = [
    'O',  # Outside
    'B-PER',  # Person
    'I-PER',
    'B-ORG',  # Organization
    'I-ORG',
    'B-LOC',  # Location
    'I-LOC'
]

label2id = {tag: i for i, tag in enumerate(tags)}
id2label = {i: tag for i, tag in enumerate(tags)}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(tags),
    id2label=id2label,
    label2id=label2id
)


# 2. 数据预处理函数
def align_labels_with_tokens(labels, word_ids):
    """将标签与tokenizer分词的token对齐"""
    new_labels = []
    current_word = None

    for word_id in word_ids:
        if word_id is None:
            # 特殊token（如[CLS], [SEP]）设置为-100，在计算损失时忽略
            new_labels.append(-100)
        elif word_id != current_word:
            # 当前word_id对应的新词开始
            current_word = word_id
            label = labels[word_id]
            new_labels.append(label)
        else:
            # 当前token是同一个词的一部分
            label = labels[word_id]
            # 如果是B-标签，则转换为I-标签
            if label % 2 == 1:  # B-标签的ID都是奇数
                label += 1  # 转换为对应的I-标签
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    """处理数据集中的每个样本"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512
    )

    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# 3. 加载并预处理数据集
def load_and_preprocess_data():
    # 加载MSRA NER数据集
    ds = load_dataset("doushabao4766/msra_ner_k_V3")

    # 预处理数据集
    tokenized_ds = ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=ds["train"].column_names
    )

    # 转换为PyTorch格式
    tokenized_ds.set_format("torch")
    return tokenized_ds


# 4. 评估指标
def compute_metrics(p):
    seqeval = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除忽略的标签（-100）
    true_predictions = [
        [tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tags[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# 5. 训练函数
def train_model():
    # 加载数据
    dataset = load_and_preprocess_data()

    # 训练参数
    training_args = TrainingArguments(
        output_dir="msra_ner_model",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard",
        save_total_limit=3,
    )

    # 数据收集器
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    print("Starting training...")
    trainer.train()
    print("Training completed!")

    # 保存模型
    trainer.save_model("msra_ner_model")
    tokenizer.save_pretrained("msra_ner_model")

    return trainer


# 6. 推理函数
class NERPredictor:
    def __init__(self, model_path=None):
        if model_path:
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = model
            self.tokenizer = tokenizer
        self.model.eval()

    def predict(self, text):
        # 中文需要先分词（简单按字符分）
        tokens = list(text)

        # Tokenize输入
        inputs = self.tokenizer(
            tokens,
            return_tensors="pt",
            truncation=True,
            is_split_into_words=True
        )

        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取预测标签
        predictions = torch.argmax(outputs.logits, dim=2)
        predicted_tags = [tags[i] for i in predictions[0].tolist()]

        # 对齐token和标签
        word_ids = inputs.word_ids()

        # 提取实体
        entities = []
        current_entity = None

        for token, tag, word_id in zip(tokens, predicted_tags, word_ids):
            if word_id is None:  # 跳过特殊token
                continue

            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'entity': tag[2:],
                    'start': word_id,
                    'end': word_id + 1,
                    'text': tokens[word_id]
                }
            elif tag.startswith('I-') and current_entity:
                if tag[2:] == current_entity['entity']:
                    current_entity['end'] = word_id + 1
                    current_entity['text'] = ''.join(tokens[current_entity['start']:current_entity['end']])
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities


# 7. 主程序
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Text to predict entities')
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.predict:
        predictor = NERPredictor("msra_ner_model")
        entities = predictor.predict(args.predict)
        print(f"Text: {args.predict}")
        print("Entities:")
        for ent in entities:
            print(f"{ent['entity']}: {ent['text']} (position: {ent['start']}-{ent['end']})")
    else:
        print("Please specify either --train to train the model or --predict <text> to predict entities")