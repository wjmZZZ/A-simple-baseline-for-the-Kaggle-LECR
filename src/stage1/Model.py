from dataclasses import dataclass

import numpy as np
from transformers import AutoModel, AutoConfig
from transformers.utils import ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import cos_sim

class ContextPooler(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = F.gelu(pooled_output)

        return pooled_output   
    
    
@dataclass
class SBertOutput(ModelOutput):
    """
    Used for SBert Model Output
    """
    loss: torch.Tensor = None
    pooled_embeddings: torch.Tensor = None
        
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, token_embeddings, attention_mask):
        """
        Average the output embeddings using the attention mask 
        to ignore certain tokens.
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    

    
class SBert(nn.Module):
    """
    Basic SBert wrapper. Gets output embeddings and averages them, taking into account the mask.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states = True)


        self.backbone = AutoModel.from_pretrained(cfg.model)

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


    def forward(self, input_ids, attention_mask, labels=None, **kwargs):

        outputs = self.backbone(input_ids, attention_mask=attention_mask, **kwargs)

        return SBertOutput(
            loss=None, # loss is calculated in `compute_loss`, but needed here as a placeholder
            pooled_embeddings=self.pool(outputs[0], attention_mask),
        )

# Basically the same as this: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
class MultipleNegativesRankingLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings_a, embeddings_b, labels=None):
        """
        Compute similarity between `a` and `b`.
        Labels have the index of the row number at each row. 
        This indicates that `a_i` and `b_j` have high similarity 
        when `i==j` and low similarity when `i!=j`.
        """
        # bs, dims x dims, bs  -> bs, bs
        similarity_scores = (
            cos_sim(embeddings_a, embeddings_b) * 20.0
        )  # Not too sure why to scale it by 20: https://github.com/UKPLab/sentence-transformers/blob/b86eec31cf0a102ad786ba1ff31bfeb4998d3ca5/sentence_transformers/losses/MultipleNegativesRankingLoss.py#L57
        
        # 一个batch
        labels = torch.tensor(
            range(len(similarity_scores)),
            dtype=torch.long,
            device=similarity_scores.device,
        )  # Example a[i] should match with b[i]

        return self.loss_function(similarity_scores, labels)