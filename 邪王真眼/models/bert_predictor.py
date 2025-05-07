import torch
import torch.nn as nn

from transformers import BertModel, BertConfig
from models.losses.loss_functions import get_loss_function


class BertPredictor(nn.Module):
    def __init__(self, cfg: dict):
        super(BertPredictor, self).__init__()
        vocab_size = cfg.get("vocab_size")
        num_layers = cfg.get("num_layers")
        dropout = cfg.get("dropout")

        bert_cfg = BertConfig.from_pretrained(
            "bert-base-chinese",
            num_hidden_layers=num_layers,
            hidden_dropout_prob=dropout,
            output_hidden_states=False,
            return_dict=True
        )
        self.bert = BertModel.from_pretrained("bert-base-chinese", config=bert_cfg)
        self.predict = nn.Linear(768, vocab_size)
        self.pad_token_id = cfg.get("pad_token_id")
        self.max_length = cfg.get("max_length")
        cfg['ignore_index'] = self.pad_token_id
        self.loss_fn = get_loss_function(cfg)
    
    def forward(self, x, mask, tag=None):
        batch_size = x.size(0)
        input_ids = x[:, :1]
        attention_mask = mask[:, :1]
        all_predictions = []
        
        for step in range(self.max_length):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            logits = self.predict(last_hidden[:, -1, :])
            all_predictions.append(logits.unsqueeze(1))
            
            if tag is not None and step < tag.size(1) - 1:
                next_token = tag[:, step + 1]
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            if (next_token == self.pad_token_id).all():
                break
                
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=x.device)], dim=1)
        
        all_predictions = torch.cat(all_predictions, dim=1)

        if all_predictions.size(1) < self.max_length:
            padding_length = self.max_length - all_predictions.size(1)
            padding = torch.zeros(batch_size, padding_length, all_predictions.size(2), 
                                device=all_predictions.device)
            all_predictions = torch.cat([all_predictions, padding], dim=1)
        
        out = all_predictions.permute(0, 2, 1).contiguous()

        output = {'out': out}
        if tag is not None:
            loss = self.loss_fn(out, tag)
            output['loss'] = loss

        return output