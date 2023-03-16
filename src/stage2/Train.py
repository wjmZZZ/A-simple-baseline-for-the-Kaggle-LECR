import gc
import os
import time
import tqdm
import wandb
import torch
import numpy as np
from scipy.special import softmax
from transformers import AdamW, AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

#from Main import LOGGER
from Config import CFG
from Model import CustomModel
from Dataset import CustomDataset
from Utils import LOGGER, AverageMeter, get_best_threshold, get_logger, get_score, get_vram, print_trick, timeSince



def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device,
             valid_loader, valid_folds, best_score, best_threshold):
    
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0

    for step, inputs in enumerate(train_loader):
        ids = inputs['input_ids'].to(device, dtype=torch.long)
        mask = inputs['attention_mask'].to(device, dtype=torch.long)
        labels = inputs['label'].to(device, dtype=torch.float)# float

        batch_size = labels.size(0)
        # 自动混合精度训练
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(ids, mask)

        loss = criterion(y_preds.view(-1), labels)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # Reset the gradients to None 
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
 
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                #   'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                        #   grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
            
        if step % (CFG.print_freq * 2) == 0:
            get_vram()

        if CFG.wandb:
            wandb.log({f"[fold{fold}] loss": losses.val,
                       f"[fold{fold}] lr": scheduler.get_lr()[0]})

        if (step+1) % CFG.val_freq == 0 or step == (len(train_loader)-1): 
            gc.collect()
            torch.cuda.empty_cache()
            avg_val_loss, valid_true, valid_pred = valid_fn(valid_loader, 
                                                            model, 
                                                            criterion, 
                                                            device
                                                            )    
            # np.save(f'valid_true.npy',valid_true)
            # np.save(f'valid_pred.npy',valid_pred)
            score, threshold = get_best_threshold(valid_folds, valid_pred, CFG)   # Evaluation


            if score > best_score: # & epoch >= CFG.epochs-2:
                LOGGER.info(f'  ======  Best Score {best_score:.5f} updating {score:.5f}, threshold {threshold}  ======\n')
                best_score = score
                best_threshold = threshold
                #if epoch >= CFG.epochs-2:  # epoch start at 0
                torch.save({
                            'model': model.state_dict(), 
                            'predictions': valid_pred
                            }, 
                            f"{CFG.output_dir}{CFG.model.split('/')[-1]}_fold{fold}.pth"
                        )
                val_predictions = valid_pred
                
            else:
                val_predictions = valid_pred
                LOGGER.info(f'  Score: {best_score:.5f} ')

    gc.collect()
    torch.cuda.empty_cache()
    return losses.avg,  avg_val_loss, val_predictions , best_score, best_threshold


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    # model.eval()
    valid_true = []
    valid_pred = []
    preds = []
    gt = []
    start = end = time.time()
    with torch.no_grad():
        for step, inputs in enumerate(valid_loader):
            ids = inputs['input_ids'].to(device, dtype=torch.long)
            mask = inputs['attention_mask'].to(device, dtype=torch.long)
            labels = inputs['label'].to(device, dtype=torch.float)
            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                y_preds = model(ids, mask)
            loss = criterion(y_preds.view(-1), labels)
            if CFG.gradient_accumulation_steps > 1:
                loss = loss / CFG.gradient_accumulation_steps

            losses.update(loss.item(), batch_size)
            # gt = labels.cpu()
            gt.append(labels.cpu())
            preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
            # print(f'preds: {preds}')

            end = time.time()
            if step+1 % CFG.val_print_freq == 0 or step == (len(valid_loader)-1):
                print('  EVAL: [{0}/{1}] '
                        'Elapsed {remain:s} '
                        'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                        .format(step, len(valid_loader),
                                loss=losses,
                                remain=timeSince(start, float(step+1)/len(valid_loader))))
       
        valid_true = np.concatenate(gt, axis = 0)
        valid_pred = np.concatenate(preds, axis = 0)

    gc.collect()
    torch.cuda.empty_cache()
    return losses.avg ,valid_true, valid_pred


