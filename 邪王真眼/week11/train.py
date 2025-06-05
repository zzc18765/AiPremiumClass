import torch
import seqeval
import evaluate
import numpy as np

from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification


def data_process(batch):
    tokenized = tokenizer(
        batch['tokens'], 
        is_split_into_words=True,
        truncation=True, 
        max_length=512, 
        padding='max_length',
        return_tensors="pt",
    )
    
    labels = torch.full(tokenized['input_ids'].shape, -100, dtype=torch.long)
    
    for i in range(len(batch['tokens'])):
        word_ids = tokenized.word_ids(batch_index=i)
        
        current_token_idx = -1
        for pos, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx != current_token_idx:
                    current_token_idx = word_idx
                    labels[i, pos] = batch['ner_tags'][i][word_idx]
                else:
                    labels[i, pos] = batch['ner_tags'][i][word_idx]
    
    tokenized['labels'] = labels
    return tokenized


def compute_metric(result):
    seqeval = evaluate.load('seqeval')
    predicts, labels = result
    predicts = np.argmax(predicts, axis=2)
    
    predicts = [[id2label[p] for p,l in zip(ps,ls) if l != -100]
                 for ps,ls in zip(predicts,labels)]
    labels = [[id2label[l] for p,l in zip(ps,ls) if l != -100]
                 for ps,ls in zip(predicts,labels)]
    results = seqeval.compute(predictions=predicts, references=labels)
    return results


if __name__ == "__main__":
    # cfg
    output_dir = "./邪王真眼/week11/result"

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')

    # dataset
    ds = load_dataset("doushabao4766/msra_ner_k_V3")
    
    label_names = ds['train'].features['ner_tags'].feature.names
    id2label = {str(i): name for i, name in enumerate(label_names)}
    label2id = {name: str(i) for i, name in enumerate(label_names)}

    ds = ds.map(data_process, batched=True)
    ds.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    # model
    model = AutoModelForTokenClassification.from_pretrained('google-bert/bert-base-chinese',
                                                            num_labels=len(label_names),
                                                            id2label=id2label,
                                                            label2id=label2id)

    # trainer
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs = 3,
        save_safetensors=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        report_to='tensorboard',
        eval_strategy="epoch",
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model,
        args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        data_collator=data_collator,
        compute_metrics=compute_metric
    )

    # 火车！
    trainer.train()

    result = trainer.predict(ds['test'])

    print(result)
