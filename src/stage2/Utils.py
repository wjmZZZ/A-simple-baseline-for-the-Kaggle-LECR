import os
import math
import time
import torch
import random
import numpy as np
import pandas as pd 
import torch.nn as nn

from transformers import  get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import  AutoTokenizer
from Config import CFG

from torch.optim import AdamW

# ====================================================
# log
# ====================================================
def get_logger(filename=CFG.output_dir+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

if not os.path.exists(CFG.output_dir):
    os.makedirs(CFG.output_dir)

####
LOGGER = get_logger()

def print_line():
    prefix, unit, suffix = "#", "--", "#"
    print(prefix + unit*50 + suffix)
def log_line():
    prefix, unit, suffix = "#", "--", "#"
    LOGGER.info(prefix + unit*50 + suffix)

# =========================================================================================
# Seed everything for deterministic results
# =========================================================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def Print_Parameter(obj): 
  LOGGER.info('\n************************************\n')
  LOGGER.info('\n'.join(['%s : %s' % item for item in obj.__dict__.items() if '__' not in item[0]]))
  LOGGER.info('\n************************************\n')


def print_trick(config, LOGGER):
    LOGGER.info(f"\n****** current version {config.version}!! ******\n")

import nvidia_smi
def get_vram():
    """Prints the total, available, and used VRAM on your machine.
    Raises an error if a NVIDIA GPU is not detected.
    """

    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("  GPU {}: {},  Memory : ({:.2f}% free): {:.2f} (total), {:.2f} (free), {:.2f} (used) <<<"
              .format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, 
                      info.total/(1024 ** 3), info.free/(1024 ** 3), info.used/(1024 ** 3)))

    nvidia_smi.nvmlShutdown()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


# =========================================================================================
# Get best threshold
# =========================================================================================
import numpy as np
from numba import njit
import polars as pl


# def get_best_threshold(valid_data, valid_predictions, cfg):
#     best_score = 0
#     best_threshold = None
#     correlations = pl.read_csv(cfg.data_dir + 'correlations.csv')

#     # Use numpy vectorization instead of for loop
#     thresholds = np.arange(0.001, 0.1, 0.001)
#     valid_data = pl.from_pandas(valid_data)
#     for i, thres in enumerate(thresholds):
#         values = np.where(valid_predictions > thres, 1, 0)
#         # Assign the list to a new column 'predictions'
#         valid_data = valid_data.with_columns(pl.lit(values).alias('predictions'))
#         # valid_data['predictions'] = np.where(valid_predictions > thres, 1, 0)

#         valid_1 = valid_data.filter(pl.col('predictions') == 1)
#         valid_1 = valid_1.groupby('topic_id').agg(pl.col('pred_content_ids').unique())
#         valid_1 = valid_1.with_columns(pl.col('pred_content_ids').apply(lambda x: ' '.join(x)).alias('predictions'))
                
#         valid_0 = pd.Series(valid_data['topic_id'].unique())
#         valid_0 = valid_0[~valid_0.isin(valid_1['topic_id'])]
#         valid_0 = pl.DataFrame({'topic_id': valid_data['topic_id'].unique(), 'predictions': ""})

#         valid_r = pl.concat([valid_1, valid_0], how = 'diagonal')
#         valid_r = valid_r.join(correlations, on='topic_id', how='left')
        
#         score = get_score(valid_r['content_ids'].to_pandas(), valid_r['predictions'].to_pandas())

#         if score > best_score:
#             best_score = score
#             best_threshold = thres

#     return best_score, best_threshold

def get_best_threshold(valid_data, val_predictions, cfg):
    best_score = 0
    best_threshold = None
    # valid_data 5列： topics_ids  content_ids	title1	title2	target
    correlations = pd.read_csv(cfg.data_dir + 'correlations.csv')  # 2列  topic_id	content_ids
    for thres in np.arange(0.01, 0.2, 0.01):
        # 验证集里的标签是和预测一一对应的，所以这里可以用阈值来限定
        valid_data['predictions'] = np.where(val_predictions > thres, 1, 0)  # 预测大于阈值，就分类为1（假设是）
        valid_1 = valid_data[valid_data['predictions'] == 1]  # 找到预测为1的content id
        valid_1 = valid_1.groupby(['topic_id'])['pred_content_ids'].unique().reset_index() # 去重
        valid_1['predictions'] = valid_1['pred_content_ids'].apply(lambda x: ' '.join(x))  # 格式统一
        # valid_1.columns = ['topic_id', 'predictions']

        valid_0 = pd.Series(valid_data['topic_id'].unique())  # 验证集中所有的去重后的主题id
        valid_0 = valid_0[~valid_0.isin(valid_1['topic_id'])]  # 找出不在预测中的主题id， 说明这部分没有预测，则content ids的预测为空
        valid_0 = pd.DataFrame({'topic_id': valid_0.values, 'predictions': ""})
        valid_r = pd.concat([valid_1, valid_0], axis = 0, ignore_index = True)
        valid_r = valid_r.merge(correlations, how = 'left', on = 'topic_id')
        
        score = get_score(valid_r['content_ids'], valid_r['predictions'])
        # LOGGER.info(f'  This validation score: {score:.5f} ')
        if score > best_score:
            best_score = score
            best_threshold = thres
    return best_score, best_threshold




# =========================================================================================
# F2 score metric
# =========================================================================================
def get_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)



# ====================================================
# optimizer
# ====================================================
def get_optimizer_params(cfg, model, encoder_lr, decoder_lr, weight_decay=0.0):
    # param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias"]
    if cfg.layerwise_learning_rate_decay == 1.0:
        optimizer_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            'lr': encoder_lr, 'weight_decay': weight_decay, 'initial_lr': encoder_lr},
            {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            'lr': encoder_lr, 'weight_decay': 0.0, 'initial_lr': encoder_lr},
            
            # {'params': [p for n, p in model.lstm.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            # 'lr': 2e-4, 'weight_decay': 0.0, 'initial_lr': 2e-4},
            # {'params': [p for n, p in model.gru.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            # 'lr': 2e-4, 'weight_decay': 0.0, 'initial_lr': 2e-4},

            {'params': [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            'lr': decoder_lr, 'weight_decay': 0.0, 'initial_lr': decoder_lr}
        ]
    else:
        if cfg.rnn is not None:
             optimizer_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "head" in n ], # or "pooler" in n
                "weight_decay": 0.0,
                "lr": encoder_lr,
            },
            {
                'params': [p for n, p in model.rnn.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
                'lr': encoder_lr,
            },
                            ]
        else:
            optimizer_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if "head" in n or "pooler" in n], # 
                        "weight_decay": 0.0,
                        "lr": encoder_lr,
                    }
                            ]
        # initialize lrs for every layer
        
        # num_layers = model.config.num_hidden_layers
        layers = [getattr(model, 'backbone').embeddings] + list(getattr(model, 'backbone').encoder.layer)
        layers.reverse()

        lr = encoder_lr
        for layer in layers[cfg.num_reinit_layers:]: # [cfg.num_reinit_layers:]
            lr *= cfg.layerwise_learning_rate_decay
            optimizer_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": cfg.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]

        if cfg.num_reinit_layers>0:
            for layer in layers[:cfg.num_reinit_layers]:
                optimizer_parameters += [
                    {
                        "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": cfg.weight_decay,
                        "lr": decoder_lr,
                    },
                    {
                        "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                        "lr": decoder_lr,
                    },

                ]
        
    return optimizer_parameters


# ====================================================
# scheduler
# ====================================================
def get_scheduler(cfg, optimizer, num_train_steps):
    num_warmup_steps = int(num_train_steps * cfg.warmp_ratio)  # 这里可设置比例
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles,
        )
    return scheduler

