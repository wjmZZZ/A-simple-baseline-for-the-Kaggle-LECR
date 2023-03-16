import gc
import os
import time
import pandas as pd
import numpy as np
import json
from itertools import chain
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from datasets import set_caching_enabled

from Config import CFG
from Utils import LOGGER, Print_Parameter, RecallAtK, compute_cv_metric, get_optimizer_params, get_scheduler, log_line, post_process, print_line, seed_everything
from Dataset import MNRCollator, cv,  get_data, uns_dataset
from Model import MultipleNegativesRankingLoss, SBert
from Train import train_fn, valid_fn
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# set_caching_enabled(False)


# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold, device):
    LOGGER.info(f"\n========== fold: {fold} training ==========")
 
    # ====================================================
    # Data Prepare
    # ====================================================
    train_folds = folds[folds['kfold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['kfold'] == fold].reset_index(drop=True)

    valid_folds.drop(valid_folds[valid_folds["category"] == "source"].index, inplace=True)
    print(f'len(valid_folds): {len(valid_folds)}')

    metric = RecallAtK(valid_folds, k=CFG.top_k)


    # ====================================================
    # Dataset
    # ====================================================
    train_dataset = uns_dataset(train_folds, CFG)
    valid_dataset = uns_dataset(valid_folds, CFG)
    LOGGER.info(f"{len(train_dataset)} training examples, {len(valid_dataset)} validation examples")

    # ====================================================
    # DataLoader
    # ====================================================   
    collate_fn = MNRCollator(CFG.tokenizer, pad_to_multiple_of= 8, max_length= CFG.max_len)

    train_params = {'batch_size': CFG.batch_size,
                    'shuffle': True,
                    'collate_fn' : collate_fn,
                    'num_workers': CFG.num_workers, 
                    'pin_memory': True, 
                    'drop_last': True
                    }
    valid_params = {'batch_size': int(CFG.batch_size * 2),
                    'shuffle': False,
                    'collate_fn' : collate_fn,
                    'num_workers': CFG.num_workers, 
                    'pin_memory': True, 
                    'drop_last': False
                    }
    train_loader = DataLoader(train_dataset, **train_params)
    valid_loader = DataLoader(valid_dataset, **valid_params)


    # ====================================================
    # step
    # ====================================================
    # print(int(len(train_dataset) / CFG.batch_size / CFG.gradient_accumulation_steps * CFG.epochs))
    num_train_steps = len(train_loader) * CFG.epochs / CFG.gradient_accumulation_steps
    epoch_steps = int(num_train_steps / CFG.epochs)

    if not CFG.debug:
        CFG.print_freq = epoch_steps // CFG.print_each_epoch
        CFG.val_print_freq = len(valid_loader) // CFG.print_each_epoch
        CFG.val_freq = epoch_steps // CFG.val_each_epoch 
    else:
        CFG.print_freq = epoch_steps // CFG.print_each_epoch
        CFG.val_print_freq = len(valid_loader) // CFG.print_each_epoch
        CFG.val_freq = epoch_steps // 1

    LOGGER.info(f'\nnum_train_steps {num_train_steps}, one_epoch_steps {epoch_steps}, val_freq:{CFG.val_freq} \n')


    # ====================================================
    # model & optimizer & scheduler
    # ====================================================
    model = SBert(CFG)    
    model.to(device)
    
    optimizer_parameters = get_optimizer_params(CFG,
                                                model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay
                                                )
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr,  eps=CFG.eps, betas=CFG.betas)

    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    loss_fc = MultipleNegativesRankingLoss()
    best_score = 0
    
    if not CFG.debug:
        untrained_scores, _, _ = valid_fn(model, valid_loader, metric, device)
        LOGGER.info(f" ****** Before training | Eval scores ******")
        for key,value in untrained_scores.items():
            LOGGER.info(f'   {key: <10} = {value:.5f}')
        torch.cuda.empty_cache()
        gc.collect()
    for epoch in range(CFG.epochs): 
        start_time = time.time()
        avg_loss, best_score , scores = train_fn(fold, 
                                                epoch, 
                                                train_loader, 
                                                valid_loader, 
                                                model, 
                                                loss_fc, 
                                                metric, 
                                                optimizer, 
                                                scheduler, 
                                                epoch_steps, 
                                                device, 
                                                best_score)
        
        elapsed = time.time() - start_time
        
        log_line()
        LOGGER.info(f' == Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s ==')
        LOGGER.info(f' == Epoch {epoch+1} - best_score: {best_score:.4f} ==')
        log_line()
    
    topic_ids = torch.load(f"{CFG.output_dir}{CFG.model.split('/')[-1]}_fold{fold}.pth", 
                            map_location=torch.device('cpu'))['topic_ids']
    pred_content_ids = torch.load(f"{CFG.output_dir}{CFG.model.split('/')[-1]}_fold{fold}.pth", 
                            map_location=torch.device('cpu'))['pred_content_ids']
    
    oof = pd.DataFrame()
    oof['topic_id'] = topic_ids
    oof['pred_content_ids'] = pred_content_ids
    corr = pd.read_csv(f'{CFG.data_dir}correlations.csv')
    corr["content_ids"] = [x.split() for x in corr["content_ids"]]
    oof = oof.merge(corr, on='topic_id')
    # grouped = valid_folds[["topic_id", "content_id"]].groupby("topic_id").agg(list)  # correlation.csv的形式，格式不同，其实就是当前验证集的部分
    # true_content_ids = grouped.loc[topic_ids].reset_index()["content_id"]
    # oof['content_ids'] = true_content_ids
    oof['kfold'] = fold

    torch.cuda.empty_cache()
    gc.collect()

    return oof

# ====================================================
# main
# ====================================================
if __name__ == '__main__':
    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)
    if not os.path.exists(CFG.stage1_data_dir):
        os.makedirs(CFG.stage1_data_dir)

    seed_everything(seed=CFG.seed)   # 设置种子!!!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Print_Parameter(CFG)
    # print_trick(CFG, LOGGER)
    print(f'device: {device}')

    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    topic_df, content_df, corr_df, exploded = get_data(CFG.data_dir)
    train = cv(topic_df, content_df, exploded, CFG.n_fold)

    sep = CFG.tokenizer.sep_token
    train['t_text'] =  train['topic_title'] 
    train['c_text'] =  train['content_title'] 

    LOGGER.info(train.kfold.value_counts())

    if CFG.debug:
        print(f'\n!!!!!!! This time it is debug !!!!!!!\n')
        CFG.epochs = 1
        CFG.val_each_epoch = 0
        # CFG.trn_fold = [0,1]
        # train = train.shuffle().select(range(5000))

    history = {}
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(train, fold, device)
            oof_df = pd.concat([oof_df, _oof_df])
            log_line()
            LOGGER.info(f"\n========== Fold: {fold} Best Result ==========")
            cv_score = compute_cv_metric(_oof_df, CFG.top_k)
            for key,value in cv_score.items():
                LOGGER.info(f'   {key: <10} = {value:.5f}')
            history[f'fold{fold}'] = cv_score
            log_line()
            
    LOGGER.info(f"\n========== OOF CV ==========")
    oof_cv = compute_cv_metric(oof_df, CFG.top_k)
    history[f'oof_cv'] = oof_cv
    for key,value in oof_cv.items():
        LOGGER.info(f'   {key: <10} = {value:.5f}')
        
    LOGGER.info(f"\n========== {CFG.version} Done!! ==========\n")
    LOGGER.info('\n'.join(['%s : %s' % item for item in history.items()]))

    oof_df.to_csv(CFG.output_dir+'oof_df.csv')
    LOGGER.info("\n========== oof_df.csv Save Successful !! ==========\n")

    for top_k in CFG.top_k:
        oof_df = pd.read_csv(CFG.output_dir+'oof_df.csv')
        train = post_process(oof_df, topic_df, content_df, top_k)
        train.to_csv(CFG.output_dir+f'train_top{top_k}.csv')
        LOGGER.info(f'\n train_top{top_k} save successful !!! \n')

    LOGGER.info("\n========== train.csv Save Successful !! ==========\n")



    if CFG.wandb:
        wandb.finish()