import os
import math
import time
import torch
import random
import numpy as np
import pandas as pd 
import torch.nn as nn
import heapq
from dataclasses import dataclass
from typing import Callable
from transformers import  get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import  AutoTokenizer
import evaluate
import datasets
from torch.optim import AdamW
from tqdm.auto import tqdm


from Config import CFG
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


class RecallAtK(evaluate.Metric):
    """Computes recall@k for a given k."""

    def __init__(self, val_df, k=100, filter_by_lang=True, **kwargs):
        super().__init__(**kwargs)

        self.val_df = val_df
        self.k = k
        self.filter_by_lang = filter_by_lang

    def _info(self):
        return evaluate.MetricInfo(
            description="metric for MNR",
            citation="No citation",
            homepage="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions_a": datasets.Sequence(datasets.Value("float32")),
                    "predictions_b": datasets.Sequence(datasets.Value("float32")),
                }
            ),
        )

    def _compute(self, predictions_a, predictions_b):
        label_ids = None
        eval_predictions = ((np.array(predictions_a), np.array(predictions_b)), label_ids)
        return compute_metrics(eval_predictions, self.val_df, self.k, self.filter_by_lang)


def cos_sim(a, b):
    # From https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L31
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0) # 在第一个维度上增加一维 （1，） -> （1,1）

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


# 返回的格式 ： [[] for _ in range(len(query_embeddings))]
# From: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L204
def semantic_search(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cos_sim,
):
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.
    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]  # 对每个topic建立一个空列表，最后要选出来top k个候选


    '''
    将content作为候选content库，这个循环通过不断遍历topic，对所有的候选content计算相似度，来得到最终的top K个候选
    最终的结果存在queries_result_list中，元素是列表，即每个topic对应的topK个候选，这topK个候选存储格式为 (score, corpus_id)
    相似度分数，对应在content库中的索引
    '''
    # 一次计算query_chunk_size个topic emb和corpus_chunk_size个content emb之间的相似度
    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):  
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities  返回一个矩阵 res[i][j]  = cos_sim(a[i], b[j])  维度为： [query_chunk_size, corpus_chunk_size]
            # 行为topic emb的个数，列为content emb的个数，即每个topic与候选库里的所有content的相似度
            cos_scores = score_function(
                query_embeddings[query_start_idx : query_start_idx + query_chunk_size],
                corpus_embeddings[
                    corpus_start_idx : corpus_start_idx + corpus_chunk_size
                ],
            )

            # Get top-k scores  得到的维度都是 [query_chunk_size, top k]
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k, len(cos_scores[0])),
                dim=1, # 指定在哪个维度上排序， 默认是最后一个维度，这里是对content 维度排序 ，dim=1表示按照行求 topn
                largest=True, # 按照大到小排序
                sorted=False, # 按照顺序返回
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            # 因为是分块计算相似度矩阵，因此每次循环对应的索引需要加上之前的，得到真正的索引
            for query_itr in range(len(cos_scores)): # len(cos_scores) = query_chunk_size
                # [query_chunk_size, top k]
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        # heapq.heappush(heap, item)， 将 item 的值加入 heap 中，保持堆的不变性
                        # 这里是将每个主题对应的content列表看作一个堆结构，然后传入topK相似度分数和这些topK对应的正常索引
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        # 将 item 放入堆中，然后弹出并返回 heap 的最小元素， 如果已经得到了topk，那么就将相似度最小的候选content弹出
                        heapq.heappushpop(
                            queries_result_list[query_id], (score, corpus_id)
                        )

    # change the data format and sort
    for query_id in range(len(queries_result_list)):  # 全部的topic个数
        for doc_itr in range(len(queries_result_list[query_id])): # 当前topic下对应的top K个候选
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {
                "corpus_id": corpus_id,
                "score": score,
            }
        queries_result_list[query_id] = sorted(
            queries_result_list[query_id], key=lambda x: x["score"], reverse=True 
        )

    return queries_result_list
   
def get_topk_preds(topic_embeddings, content_embeddings, df, k=100, return_idxs=True):
    """
    df has the same number of rows as topic_embeddings and content_embeddings.
    A bunch of the topic embeddings are duplicates, so we'll dedupe before finding 
    nearest neighbors.
    一些topic embedding是重复的，所以我们要在寻找近邻之前进行剔除。  因为 explode 
    Returns tuple (prediction content ids, topic ids)
    
        prediction content ids is a list of lists. The outer list has the same number of elements as unique topic
        ids. The inner list contains content ids and the length is equal to k (num nearest neighbors).
    """
    
    # These idx values will be used to compare with the idx
    # values returned by `semantic_search` to calculate recall.
    df["idx"] = list(range(len(content_embeddings)))  # 原始索引编号
    
    deduped_topics = df[["idx", "topic_id"]].drop_duplicates("topic_id")  # 得到每个exploded的唯一的topic id
    topic_embeddings = topic_embeddings[deduped_topics["idx"]]
    device = torch.device("cuda:0")
    
    deduped_content = df[["idx", "content_id"]].drop_duplicates("content_id")  #TODO 为什么这里还要取出content id的重复
    content_ids = deduped_content.content_id.values
    content_embeddings = content_embeddings[deduped_content["idx"]]
    

    # Compare each of the topic embeddings to each of the
    # content embeddings and return a ranking for each one.
    # Works much, much faster on GPU.
    search_results = semantic_search(
        torch.tensor(topic_embeddings, device=device),
        torch.tensor(content_embeddings, device=device),
        top_k=k,
    )
    
    # `search_results` is a list of lists. The inner listhas a `dict` at each element.
    # The dict has two keys: `corpus_id` and `score`.
    all_pred_c_ids = [[content_ids[x["corpus_id"]] for x in row] for row in search_results]
    
    return all_pred_c_ids, deduped_topics["topic_id"].tolist()

def precision_score(pred_content, gt_content):
    """
    Arguments can be int (idx) or string values.
    """
    def precision(pred, gt):
        tp = len(set(pred)&set(gt))
        fp = len(set(pred)-set(gt))
        return tp/(tp+fp+1e-7)
    
    # Get a recall score for each row of the dataset
    return [precision(pred, gt) for pred, gt in zip(pred_content, gt_content)]

def recall_score(pred_content, gt_content):
    """
    Arguments can be int (idx) or string values.
    """
    def recall(pred, gt):
        tp = len(set(pred)&set(gt))
        return tp/len(set(gt))
    
    # Get a recall score for each row of the dataset
    return [recall(pred, gt) for pred, gt in zip(pred_content, gt_content)]

def mean_f2_score(precision_scores, recall_scores):
    """
    Inputs should be outputs of the `precision_score` and  `recall_score` functions.
    """
    beta = 2
    
    def f2_score(precision, recall):
        return (1+beta**2)*(precision*recall)/(beta**2*precision+recall+1e-7)
    
    return round(np.mean([f2_score(p, r) for p, r in zip(precision_scores, recall_scores)]), 5)
    


def compute_metrics(eval_predictions, val_df, k=100, filter_by_lang=True):
    """
    After creating embeddings for all of the topic and content texts,
    perform a semantic search and measure the recall@100. The model
    has not seen any of these examples before, so it should be a
    good measure of how well the model can generalize.

    Since the dataset uses the exploded view of the correlations
    (one topic with 5 contents is 5 rows), I need to deduplicate
    the topic embeddings. Then I can use the `semantic_search`
    function taken from the sentence-transformers util to
    do a cosine similarity search of the topic embeddings with all
    content embeddings. This function conveniently returns the top
    `k` indexes, which makes it easy to compare with the true indexes.
    在为所有的主题和内容文本创建嵌入后。 进行语义搜索并测量recall@100。
    该模型以前没有见过任何这些例子，所以它应该是衡量该模型的概括能力的好办法。

    由于该数据集使用的是关联的exploded view (一个有5个内容的主题是5行)，
    所以我需要重复计算 topic embedding。然后，我可以使用取自sentence transformer的`semantic_search`函数，
    对topic embedding与所有的content embedding进行余弦相似度搜索。这个函数方便地返回前`k'个索引，
    这使得它很容易与真正的索引进行比较。
    """
    
    if isinstance(k, int):
        k = [k]

    # eval_predictions is a tuple of (model_output, labels)
    # The model_output is whatever is returned by `compute_loss`
    (topic_embeddings, content_embeddings), _ = eval_predictions  # ((np.array(predictions_a), np.array(predictions_b)), label_ids)

    pred_content_ids, topic_ids = get_topk_preds(topic_embeddings, content_embeddings, val_df, k=max(k))
    
    # Filter based on language
    
    if filter_by_lang:
        content2lang = {content_id: lang for content_id, lang in val_df[["content_id", "content_language"]].values}
        topic2lang = {topic_id: lang for topic_id, lang in val_df[["topic_id", "topic_language"]].values}

        filtered_pred_content_ids = []
        for preds, topic_id in zip(pred_content_ids, topic_ids):
            topic_lang = topic2lang[topic_id]
            # 如果得到的topK候选content的语言与其对应的topic的语言相同就留下，否则丢弃
            filtered_pred_content_ids.append([c_id for c_id in preds if content2lang[c_id]==topic_lang])
            
        pred_content_ids = filtered_pred_content_ids # [[], [], [] , ,,,,]
    # topic_ids_ = np.array(topic_ids)
    # pred_content_ids_ = np.array(pred_content_ids)
    # np.save('topic_ids.npy', topic_ids_)
    # np.save('pred_content_ids.npy', pred_content_ids_)

    # print(f'pred_content_ids:\n {pred_content_ids}, len() : {pred_content_ids}')
    # Make sure true content ids are in same order as predictions
    grouped = val_df[["topic_id", "content_id"]].groupby("topic_id").agg(list)  # correlation.csv的形式，格式不同，其实就是当前验证集的部分
    true_content_ids = grouped.loc[topic_ids].reset_index()["content_id"]
    
    metrics = {}
    for kk in k: # k是一个列表
        top_preds = [row[:kk] for row in pred_content_ids]
        precisions = precision_score(top_preds, true_content_ids)
        recalls = recall_score(top_preds, true_content_ids)
        f2 = mean_f2_score(precisions, recalls)
        
        metrics[f"recall@{kk}"] = np.mean(recalls)
        metrics[f"f2@{kk}"] = f2
    
    return metrics, topic_ids, pred_content_ids


# =========================================================================================
# Get the amount of positive classes based on the total
# =========================================================================================
def compute_cv_metric(oof, top_k):
    pred_content_ids = oof['pred_content_ids'].values.tolist()
    true_content_ids = oof['content_ids'].values.tolist()
    metrics = {}
    for kk in top_k: # k是一个列表
        top_preds = [row[:kk] for row in pred_content_ids]
        precisions = precision_score(top_preds, true_content_ids)
        recalls = recall_score(top_preds, true_content_ids)
        f2 = mean_f2_score(precisions, recalls)
        metrics[f"Recall@{kk}"] = np.mean(recalls)
        metrics[f"F2@{kk}"] = f2
    return metrics


import ast
import swifter
def post_process(oof, topic_df, content_df, top_k):
    print(f'Top k : {top_k}')
    oof['pred_content_ids'] = oof['pred_content_ids'].apply(ast.literal_eval)
    oof['content_ids'] = oof['content_ids'].apply(ast.literal_eval)
    oof['pred_content_ids'] = [row[:top_k] for row in oof['pred_content_ids']]  # 每个topic只选top k个content
    oof = oof.explode("pred_content_ids").reset_index(drop=True)
    oof['target'] = oof.swifter.apply(lambda x: 1 if x.pred_content_ids in x.content_ids else 0, axis=1)
    LOGGER.info("Label counts")
    LOGGER.info(f'{oof.target.value_counts()}')
    oof = oof.merge(topic_df[['topic_id', 'topic_title', 'topic_description']], on="topic_id")
    oof = oof.merge(content_df[['content_id', 'content_title', 'content_description', 'content_text']], how = 'inner', left_on = 'pred_content_ids', right_on = 'content_id')
    return oof 