import gc
import json
from pathlib import Path
from itertools import chain
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from Model import SBert, MultipleNegativesRankingLoss
from Utils import LOGGER, AverageMeter, compute_metrics, get_vram, print_line, timeSince
from Config import CFG
from Metric import RecallAtK
from Dataset import MNRCollator


def train_fn(fold, epoch, train_loader, valid_loader, model, loss_fc, metric, optimizer, scheduler, epoch_steps, device
             ,best_score):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0

    for step, batch in enumerate(train_loader):
        ids_a = batch["input_ids_a"].to(device)
        mask_a = batch["attention_mask_a"].to(device)
        ids_b = batch["input_ids_b"].to(device)
        mask_b = batch["attention_mask_b"].to(device)
        
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            embeddings_a = model(ids_a, mask_a).pooled_embeddings
            embeddings_b = model(ids_b, mask_b,).pooled_embeddings

        loss = loss_fc(embeddings_a, embeddings_b)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        optimizer.zero_grad()   
        losses.update(loss.item(), len(batch))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

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
                          #grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
            
        if step % (CFG.print_freq * 2) == 0:
            get_vram()
            
        if (step+1) % CFG.val_freq == 0 or step == (len(train_loader)-1): 
                scores, topic_ids, pred_content_ids = valid_fn(model, valid_loader, metric, device)
                metric_score = scores[CFG.metric_to_track]
                
                
                if metric_score > best_score:
                    LOGGER.info(f'\n === Best Score {best_score:.4f} updating {metric_score:.4f} === \n')
                    LOGGER.info(f"Epoch {round(epoch+step/epoch_steps, 2)}/{CFG.epochs}| Eval scores:")
                    for key,value in scores.items():
                        LOGGER.info(f'   {key: <10} = {value:.5f}')
                    torch.save({
                                'model': model.state_dict(), 
                                'topic_ids': topic_ids,
                                'pred_content_ids':pred_content_ids
                                }, 
                                f"{CFG.output_dir}{CFG.model.split('/')[-1]}_fold{fold}.pth"
                            )
                    best_score = metric_score

    return losses.avg, best_score , scores

def valid_fn(model, loader, metric, device):
    # LOGGER.info(f'--- Evaluation --- ')
    # model.eval()
    start = end = time.time()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
        
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                embeds_a = model(ids_a, mask_a).pooled_embeddings
                embeds_b = model(ids_b, mask_b).pooled_embeddings
                metric.add_batch(predictions_a=embeds_a, predictions_b=embeds_b)
            if step+1 % CFG.val_print_freq == 0 or step == (len(loader)-1):
                print('  EVAL: [{0}/{1}] '
                        'Elapsed {remain:s} '
                        .format(step, len(loader),
                                remain=timeSince(start, float(step+1)/len(loader))))
                
    metrics, topic_ids, pred_content_ids = metric.compute()

    torch.cuda.empty_cache()
    gc.collect()
    return metrics, topic_ids, pred_content_ids