from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# 初始化DDP
def setup_ddp():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


# 加载数据集
ds = load_dataset("doushabao4766/msra_ner_k_V3")
label_list = ds["train"].features["ner_tags"].feature.names
num_labels = len(label_list)
print("标签列表:", label_list)

# 加载模型和tokenizer
model_name = "google-bert/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 在DDP中，模型需要在初始化后移动到正确的设备
def get_model():
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)}
    )
    if torch.cuda.is_available():
        model.cuda()
    if dist.is_initialized():
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    return model


# 数据预处理函数保持不变
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 应用预处理
tokenized_datasets = ds.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=ds["train"].column_names
)


# 动态学习率调度器
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 自定义Trainer类以实现动态学习率
class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # 设置学习率调度器
        num_warmup_steps = int(0.1 * num_training_steps)  # 10%的训练步数作为warmup
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )


# 训练函数
def train_model():
    # 训练参数设置
    training_args = TrainingArguments(
        output_dir="./ner_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        fp16=True,  # 启用混合精度训练
        gradient_accumulation_steps=2,  # 梯度累积
        dataloader_num_workers=4,
        logging_steps=100,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,  # DDP设置
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),  # DDP设置
    )

    # 数据收集器
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 创建模型
    model = get_model()

    # 创建Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        # compute_metrics=compute_metrics
    )

    # 训练
    trainer.train()

    # 保存模型
    if trainer.is_world_process_zero():  # 只在主进程保存
        trainer.save_model("./ner_final_model")
        tokenizer.save_pretrained("./ner_final_model")


# NERPredictor类保持不变
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
        tokens = list(text)
        inputs = self.tokenizer(
            tokens,
            return_tensors="pt",
            truncation=True,
            is_split_into_words=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)
        predicted_tags = [label_list[i] for i in predictions[0].tolist()]
        word_ids = inputs.word_ids()

        entities = []
        current_entity = None

        for token, tag, word_id in zip(tokens, predicted_tags, word_ids):
            if word_id is None:
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


# 主程序
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Text to predict entities')
    args = parser.parse_args()

    if args.train:
        # 检查是否使用DDP
        if "LOCAL_RANK" in os.environ:
            setup_ddp()
        train_model()
    elif args.predict:
        predictor = NERPredictor("./ner_final_model")
        entities = predictor.predict(args.predict)
        print(f"Text: {args.predict}")
        print("Entities:")
        for ent in entities:
            print(f"{ent['entity']}: {ent['text']} (position: {ent['start']}-{ent['end']})")
    else:
        print("Please specify either --train to train the model or --predict with text to predict entities")