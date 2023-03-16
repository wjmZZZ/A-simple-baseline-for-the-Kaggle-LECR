# ====================================================
# Model
# ====================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from Config import CFG


# =========================================================================================
# Mean pooling class
# =========================================================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
# =========================================================================================
# Model
# =========================================================================================
class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states = True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        self.backbone = AutoModel.from_pretrained(cfg.model, config = self.config)
        if self.cfg.gradient_checkpoint:
            self.backbone.gradient_checkpointing_enable()

        self.backbone.resize_token_embeddings(len(cfg.tokenizer)) 

        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self.pool = MeanPooling()
        self.head = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, input_ids, mask):
        outputs = self.backbone(input_ids, mask)

        pooled_output = outputs[1]

        last_cat = self.dropout(pooled_output)

        logits1 = self.head(self.dropout1(last_cat))
        logits2 = self.head(self.dropout2(last_cat))
        logits3 = self.head(self.dropout3(last_cat))
        logits4 = self.head(self.dropout4(last_cat))
        logits5 = self.head(self.dropout5(last_cat))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        return logits