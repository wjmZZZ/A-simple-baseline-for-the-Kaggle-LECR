import pandas as pd
import numpy as np
import torch
import json
import networkx as nx
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass
from transformers import PreTrainedTokenizer

from Config import CFG 
from Utils import LOGGER


def get_data(data_dir):
    content_df = pd.read_csv(data_dir + "content.csv")
    corr_df = pd.read_csv(data_dir + "correlations.csv")
    topic_df = pd.read_csv(data_dir + "topics.csv")

    topic_df = topic_df.rename(
        columns={
            "id": "topic_id",
            "title": "topic_title",
            "description": "topic_description",
            "language": "topic_language",
        }
    )
    content_df = content_df.rename(
        columns={
            "id": "content_id",
            "title": "content_title",
            "description": "content_description",
            "text": "content_text",
            "language": "content_language"
        }
    )

    # Fill in blanks and limit the amount of content text
    topic_df["topic_title"].fillna("No topic title", inplace=True)
    topic_df["topic_description"].fillna("No topic description", inplace=True)
    content_df["content_title"].fillna("No content title", inplace=True)
    content_df["content_description"].fillna("No content description", inplace=True)
    content_df["content_text"].fillna("No content text", inplace=True)
    content_df["content_text"] = [x[:300] for x in content_df["content_text"]]

    # `exploded` has one topic id and one content id per row
    corr_df["content_id"] = [x.split() for x in corr_df["content_ids"]]
    exploded = corr_df.explode("content_id")
    
    return topic_df, content_df, corr_df, exploded

# =========================================================================================
# CV split
# =========================================================================================
from sklearn.model_selection import KFold


def cv(topic_df, content_df, exploded, kfold):
    train = topic_df.merge(exploded, on="topic_id").merge(content_df, on="content_id")
    train.drop(['kind', 'level', 'parent', 'copyright_holder'], axis = 1, inplace = True)
    # Remove all topic ids with has_content = false
    train.drop(train[train['has_content'] == False].index, inplace=True)  

    source_train = train[train["category"] == "source"]
    train.drop(train[train["category"] == "source"].index, inplace=True)
    train = train.reset_index(drop=True)


    train["kfold"] = -1
    kf = KFold(n_splits=kfold) # 实例化（k折交叉验证）
    for fold, ( _, val_) in enumerate(kf.split(X=train, y=train['category'])):
        train.loc[val_ , "kfold"] = fold
        
    return train                   


           



def tokenize(batch, tokenizer, topic_cols, content_cols, max_length):
    """
    Tokenizes the dataset on the specific columns, truncated/padded to a max length.
    Adds the suffix "_content" to the input ids and attention mask of the content texts.

    Returns dummy labels that make the evaluation work in Trainer.
    """
    sep = tokenizer.sep_token

    # 将几段文本用分隔符隔开，'topic_title', 'topic_description' 'content_title', 'content_description','content_text'
    topic_texts = [sep.join(cols) for cols in zip(*[batch[c] for c in topic_cols])]
    content_texts = [sep.join(cols) for cols in zip(*[batch[c] for c in content_cols])]

    tokenized_topic = tokenizer(
        topic_texts, truncation=True, max_length=max_length, padding=False
    )

    tokenized_content = tokenizer(
        content_texts, truncation=True, max_length=max_length, padding=False
    )

    # Remove token_type_ids. They will just cause errors.
    if "token_type_ids" in tokenized_topic:
        del tokenized_topic["token_type_ids"]
        del tokenized_content["token_type_ids"]

    return {
        **{f"{k}_a": v for k, v in tokenized_topic.items()},
        **{f"{k}_b": v for k, v in tokenized_content.items()},
        "labels": [1] * len(tokenized_topic.input_ids), # placeholder for Trainer
    }


def get_tokenized_ds(ds, tokenizer, max_length=64):
    tokenized_ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            topic_cols=[f"topic_{c}" for c in CFG.topic_cols],
            content_cols=[f"content_{c}" for c in CFG.content_cols],
            max_length=max_length,
        ),
        remove_columns=ds.column_names,
        num_proc=CFG.num_workers,
    )

    return tokenized_ds

@dataclass
class MNRCollator:
    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: int = 8
    max_length: int = 512

    def __call__(self, features):

        longest_topic = max([len(x["input_ids_a"]) for x in features])
        longest_content = max([len(x["input_ids_b"]) for x in features])

        pad_token_id = self.tokenizer.pad_token_id

        input_ids_topic = [
            x["input_ids_a"]
            + [pad_token_id]
            * (min(longest_topic, self.max_length) - len(x["input_ids_a"]))
            for x in features
        ]
        attention_mask_topic = [
            x["attention_mask_a"]
            + [0] * (min(longest_topic, self.max_length) - len(x["attention_mask_a"]))
            for x in features
        ]

        input_ids_content = [
            x["input_ids_b"]
            + [pad_token_id]
            * (min(longest_content, self.max_length) - len(x["input_ids_b"]))
            for x in features
        ]
        attention_mask_content = [
            x["attention_mask_b"]
            + [0]
            * (min(longest_content, self.max_length) - len(x["attention_mask_b"]))
            for x in features
        ]

        return {
            "input_ids_a": torch.tensor(input_ids_topic),
            "attention_mask_a": torch.tensor(attention_mask_topic),
            "input_ids_b": torch.tensor(input_ids_content),
            "attention_mask_b": torch.tensor(attention_mask_content),
        }




# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_uns_input(topic_texts, content_texts, cfg, max_len):
    tokenized_topic = cfg.tokenizer(
                                topic_texts, 
                                truncation=True, 
                                max_length=max_len, 
                                padding=False
                            )

    tokenized_content = cfg.tokenizer(
                                content_texts, 
                                truncation=True, 
                                max_length=max_len, 
                                padding=False
                            )

    # Remove token_type_ids. They will just cause errors.
    if "token_type_ids" in tokenized_topic:
        del tokenized_topic["token_type_ids"]
        del tokenized_content["token_type_ids"]

    return {
        **{f"{k}_a": v for k, v in tokenized_topic.items()},
        **{f"{k}_b": v for k, v in tokenized_content.items()},
        "labels": [1] * len(tokenized_topic.input_ids), # placeholder for Trainer
    }


# =========================================================================================
# Unsupervised dataset
# =========================================================================================
class uns_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.max_len = cfg.max_len
        self.t_text = df['t_text'].values
        self.c_text = df['c_text'].values
    def __len__(self):
        return len(self.t_text)
    def __getitem__(self, index):
        output = prepare_uns_input(
                                   self.t_text[index], 
                                   self.c_text[index], 
                                   self.cfg, 
                                   self.max_len
                                  )
        return output
        
