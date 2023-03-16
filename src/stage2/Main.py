import gc
import os
import time
import pandas as pd
import numpy as np
import warnings

from sklearn.utils import resample
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from Config import CFG
from Dataset import Collate, CrossValidation, CustomDataset, get_max_length, prepare_data, sup_dataset
from Model import CustomModel
from Train import train_fn
from Utils import LOGGER, Print_Parameter, get_best_threshold, get_optimizer_params, get_scheduler, log_line, print_trick, seed_everything

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ====================================================
# record
# ====================================================
if CFG.wandb:
    import wandb
    wandb.login()
    anony = None
    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__')) 
    run = wandb.init(project='LECR', 
                    name=CFG.model,
                    config=class2dict(CFG),
                    group=CFG.model,
                    job_type="train",
                    anonymous=anony)
else:
    wandb = None



# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold, device):
    seed_everything(seed=CFG.seed)
    LOGGER.info(f"\n========== fold: {fold} training ==========")

    # ====================================================
    # Data Prepare
    # ====================================================
    train_folds = folds[folds['kfold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['kfold'] == fold].reset_index(drop=True)
    
    # 将样本分成多数类和少数类
    df_majority = train_folds[train_folds.target == 0]
    df_minority = train_folds[train_folds.target == 1]

    # 随机从多数类中采样少数类样本数量的样本
    df_majority_downsampled = resample(df_majority, 
                            replace=False, # 不使用放回采样
                            n_samples=2*len(df_minority), # 采样数量等于少数类样本数量
                            random_state=CFG.seed) 
    train_folds = pd.concat([df_majority_downsampled, df_minority])


    LOGGER.info("train_folds Label counts")
    LOGGER.info(f'{train_folds.target.value_counts()}')
    LOGGER.info("valid_folds Label counts")
    LOGGER.info(f'{valid_folds.target.value_counts()}')
    
    valid_labels = valid_folds['target'].values
    # print(f'排序前\n valid_labels:{valid_labels}, length:{len(valid_labels)}')

    training_samples = prepare_data(train_folds, CFG, num_jobs=8)
    valid_samples = prepare_data(valid_folds, CFG, num_jobs=8)

    # ====================================================
    # Dataset
    # ====================================================
    train_dataset = CustomDataset(CFG, training_samples)
    valid_dataset = CustomDataset(CFG, valid_samples)


    # ====================================================
    # DataLoader
    # ====================================================   
    collate_fn = Collate(CFG.tokenizer)

    train_params = {'batch_size': CFG.batch_size,
                    'shuffle': True,
                    'collate_fn' : collate_fn,
                    'num_workers': CFG.num_workers, 
                    'pin_memory': True, 
                    'drop_last': True
                    }
    valid_params = {'batch_size': int(CFG.batch_size * 2),
                    'shuffle': False,
                    'collate_fn' :collate_fn,
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
        CFG.val_print_freq = len(valid_loader) // 2
        CFG.val_freq = len(train_loader) // CFG.val_each_epoch
    else:
        CFG.print_freq = epoch_steps // CFG.print_each_epoch
        CFG.val_print_freq = len(valid_loader) // CFG.print_each_epoch
        CFG.val_freq = len(train_loader) // 10

    LOGGER.info(f'\n num_train_steps {num_train_steps}, one_epoch_steps {len(train_loader)}, val_freq:{CFG.val_freq} \n')

    # ====================================================
    # model & optimizer & scheduler
    # ====================================================
    model = CustomModel(CFG)
    state = torch.load(f"{CFG.state1_path}{CFG.state1_model.split('/')[-1]}_fold{fold}.pth",
                        map_location='cuda')
    model.load_state_dict(state['model'])
    model.to(device)
    
    optimizer_parameters = get_optimizer_params(CFG,
                                                model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay
                                                )
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr,  eps=CFG.eps, betas=CFG.betas)

    scheduler = get_scheduler(CFG, optimizer, num_train_steps)



    # ====================================================
    # Train Loop
    # ====================================================
    lossfc = nn.BCEWithLogitsLoss(reduction = "mean") 
    best_score = 0.0
    best_threshold = 0.0
    for epoch in range(CFG.epochs):
        start_time = time.time()
        avg_loss, avg_val_loss, predictions, best_score, best_threshold = train_fn(fold, 
                                                                         train_loader,
                                                                         model, 
                                                                         lossfc, 
                                                                         optimizer,
                                                                         epoch, 
                                                                         scheduler, 
                                                                         device, 
                                                                         valid_loader, 
                                                                         valid_folds,
                                                                         best_score,
                                                                         best_threshold
                                                                        )
        # best_score, best_threshold = get_best_threshold(valid_folds, predictions, CFG) 
        elapsed = time.time() - start_time

        log_line()
        LOGGER.info(f'== Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'== Epoch {epoch+1} - best_score: {best_score:.4f} best_threshold: {best_threshold}')
        log_line()
        
        if CFG.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch+1, 
                       f"[fold{fold}] avg_train_loss": avg_loss, 
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] Best_score": best_score})

   

    predictions = torch.load(CFG.output_dir+f"{CFG.model.split('/')[-1]}_fold{fold}.pth", 
                             map_location=torch.device('cpu'))['predictions']


    oof = pd.DataFrame()
    oof['topic_id'] = valid_folds['topic_id']
    oof['predictions'] = predictions
    oof['label'] = valid_labels

    torch.cuda.empty_cache()
    gc.collect()

    return best_score, best_threshold, oof


# ====================================================
# main
# ====================================================
if __name__ == '__main__':

    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)
    seed_everything(seed=CFG.seed)   # 设置种子!!!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Print_Parameter(CFG)
    print_trick(CFG, LOGGER)
    print(f'device: {device}')

    # train_df, correlations_df = get_data(CFG.data_dir)
    # train_df = get_topic_data(CFG.data_dir + 'train_df_topic.csv')
    train = pd.read_csv(f'{CFG.state1_path}train_top{CFG.state1_topk}.csv')

    # train = CrossValidation(train, CFG.n_fold, CFG.seed)
    LOGGER.info(train["kfold"].value_counts())

    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    sep = CFG.tokenizer.sep_token
    train['text'] = train['topic_title'] + sep + train['content_title']
    train ['text'] = train.text.values.tolist()

    if CFG.debug:
        print('\n!!!!!!! This time it is debug !!!!!!!')
        CFG.epochs = 1
        CFG.trn_fold = [3]

    history = {}
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            fold_score, fold_threshold, _oof_df = train_loop(train, fold, device)
            oof_df = pd.concat([oof_df, _oof_df])
            log_line()
            LOGGER.info(f"========== fold: {fold} result ==========")
            LOGGER.info(f'best_score: {fold_score:.4f} best_threshold: {fold_threshold}')
            history[f'fold{fold}_score'] = fold_score
            history[f'fold{fold}_threshold'] = fold_threshold
            log_line()

    oof_df.to_csv(CFG.output_dir+'s2_oof_df.csv')
    oof_score, oof_threshold = get_best_threshold(train, oof_df['predictions'], CFG) 
    LOGGER.info(f"\n========== OOF CV ==========")
    LOGGER.info(f"oof_score : {oof_score}")
    LOGGER.info(f"oof_threshold : {oof_threshold}")

    history[f'oof_cv'] = oof_score
    history[f'oof_threshold'] = oof_threshold


    LOGGER.info(f"\n========== {CFG.version} Done!! ==========\n")
    LOGGER.info('\n'.join(['%s : %s' % item for item in history.items()]))
    log_line()    

    if CFG.wandb:
        wandb.finish()