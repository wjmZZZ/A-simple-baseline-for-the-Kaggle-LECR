
import pandas as pd 
import numpy as np
import torch
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold

from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from Config import CFG


# =========================================================================================
# Data Loading
# =========================================================================================
def get_data_stage2(data_dir, isTrain=True):
    print(f'This is supervised data, train + correlations')
    train = pd.read_csv(data_dir+'train.csv')
    train['title1'].fillna("Title does not exist", inplace = True)
    train['title2'].fillna("Title does not exist", inplace = True)
    correlations = pd.read_csv(data_dir+'correlations.csv')
    # Create feature column
    train['text'] = train['title1'] + '[SEP]' + train['title2']
    print(' ')
    print('-' * 50)
    print(f"train.shape: {train.shape}")
    print(f"correlations.shape: {correlations.shape}")
    return train, correlations

def get_data_stage1(data_dir):
    print(f'This is unsupervised data, topics + content + sample_submission')
    topics = pd.read_csv(data_dir + 'topics.csv')
    content = pd.read_csv(data_dir + 'content.csv')
    sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')
    # Merge topics with sample submission to only infer test topics
    topics = topics.merge(sample_submission, how = 'inner', left_on = 'id', right_on = 'topic_id')
    # Fillna titles
    topics['title'].fillna("", inplace = True)
    content['title'].fillna("", inplace = True)
    # Sort by title length to make inference faster
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    topics.sort_values('length', inplace = True)
    content.sort_values('length', inplace = True)
    # Drop cols
    topics.drop(['description', 'channel', 'category', 'level', 'language', 'parent', 'has_content', 'length', 'topic_id', 'content_ids'], axis = 1, inplace = True)
    content.drop(['description', 'kind', 'language', 'text', 'copyright_holder', 'license', 'length'], axis = 1, inplace = True)
    # Reset index
    topics.reset_index(drop = True, inplace = True)
    content.reset_index(drop = True, inplace = True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    return topics, content   



# =========================================================================================
# CV split
# =========================================================================================
def CrossValidation(train_df, kfold, seed):
    train_df["kfold"] = -1  # 创建一个名为 kfold 的新列，并用-1填充
    #train_df = train_df.sample(frac=1).reset_index(drop=True) # 打乱数据 
    kf = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=seed) # 实例化（k折交叉验证）
    for fold, ( _, val_) in enumerate(kf.split(train_df,  train_df['target'], train_df['topic_id'])):
        train_df.loc[val_ , "kfold"] = fold
    
    print(train_df.groupby('kfold')['target'].value_counts())
    return train_df

# =========================================================================================
# Get max length
# =========================================================================================
def get_max_length(train, cfg):
    lengths = []
    for text in tqdm(train['text'].fillna("").values, total = len(train)):
        length = len(cfg.tokenizer(text, add_special_tokens = False)['input_ids'])
        lengths.append(length)
    max_len = max(lengths) + 2 # cls & sep
    print(f"max_len: {max_len}")

# ====================================================
# Dataset
# ====================================================
def _prepare_data_helper(cfg, df, isTrain=True):
    training_samples = []
    for _, row in df.iterrows():
        text = row["text"]        
        encoded_text = cfg.tokenizer.encode_plus(
                                            text,
                                            return_tensors = None, 
                                            add_special_tokens = False, 
                                            max_length = cfg.max_len,
                                            pad_to_max_length = False,
                                            truncation = False
                                            )
        input_ids = encoded_text["input_ids"]
        if isTrain:
            inputs = {
                "input_ids": input_ids,
                "label": row['target']
                    }
        else:
             inputs = {
                "input_ids": input_ids,
                    }
        # if "token_type_ids" in encoded_text:
        #     inputs["token_type_ids"] = encoded_text["token_type_ids"]
        training_samples.append(inputs)
    return training_samples   

def prepare_data(df, cfg, num_jobs, is_train=True):
    training_samples = []

    df_splits = np.array_split(df, num_jobs)
    #train_ids = df["essay_id"].unique()
    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_data_helper)(cfg, df, is_train) for df in df_splits
    )
    for result in results:
        training_samples.extend(result)

    return training_samples

# =========================================================================================
# Custom dataset
# =========================================================================================
# class custom_dataset(Dataset):
#     def __init__(self, df, cfg):
#         self.cfg = cfg
#         self.texts = df['text'].values
#         self.labels = df['target'].values
#     def __len__(self):
#         return len(self.texts)
#     def __getitem__(self, item):
#         inputs = prepare_input(self.texts[item], self.cfg)
#         label = torch.tensor(self.labels[item], dtype = torch.float)
#         return inputs, label

class CustomDataset(Dataset):
    def __init__(self, cfg, df, isTrain=True):
        self.cfg = cfg
        self.df = df
        self.tokenizer = cfg.tokenizer
        self.isTrain = isTrain

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        ids = self.df[index]["input_ids"]
        if self.isTrain:
            label = self.df[index]["label"]

        input_ids = [self.tokenizer.cls_token_id] + ids

        if len(input_ids) > self.cfg.max_len - 1:
            input_ids = input_ids[: self.cfg.max_len - 1]

        input_ids = input_ids + [self.tokenizer.sep_token_id]
        mask = [1] * len(input_ids)
        if self.isTrain:
            return {
                "input_ids": input_ids,
                "attention_mask": mask,
                # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "label": label,
            }
        else:
             return {
                "input_ids": input_ids,
                "attention_mask": mask,
                # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            }

class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain

    def __call__(self, batch):  # sourcery skip: comprehension-to-generator, dict-literal, merge-dict-assign
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["label"] = [sample["label"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])
        #batch_max = min(batch_max, self.max_len) if self.max_len is not None else batch_max
        
        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors  避免CPU和GPU之间不必要的数据传输
        output["input_ids"] = torch.tensor(output["input_ids"])
        output["attention_mask"] = torch.tensor(output["attention_mask"])
        if self.isTrain:
            output["label"] = torch.tensor(output["label"],  dtype = torch.float)
        
        return output
    


# =========================================================================================
# Supervised dataset
# =========================================================================================
def prepare_sup_input(text, label, cfg):
    encode_inputs = cfg.tokenizer.encode_plus(
                            text, 
                            return_tensors = None,
#                             truncation=True, 
#                             max_length=max_len, 
                            add_special_tokens = True, 
                        )
    input_ids = encode_inputs["input_ids"]
    inputs = {
        "input_ids": input_ids,
        "label": label
            }
    return inputs

class sup_dataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values
        self.label = df['target'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_sup_input(self.texts[item], 
                                   self.label[item],
                                   self.cfg
                                  )
        ids = inputs[item]["input_ids"]
        label = self.df[item]["label"]

        input_ids = [CFG.tokenizer.cls_token_id] + ids

        if len(input_ids) > self.cfg.max_len - 1:
            input_ids = input_ids[: self.cfg.max_len - 1]

        input_ids = input_ids + [self.tokenizer.sep_token_id]
        mask = [1] * len(input_ids)
        return {
                "input_ids": input_ids,
                "attention_mask": mask,
                # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "label": label,
            }
