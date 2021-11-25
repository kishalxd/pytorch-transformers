import os
import gc
import copy
import time
import random
import string

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from config import Config
from dataset import ToxicDataset
from model import ToxicModel

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def metric(outputs1, outputs2):
    o1 = outputs1.cpu().detach().numpy()
    o2 = outputs2.cpu().detach().numpy()

    return np.mean(o1>o2)

def train_one_epoch(model, optimizer, scheduler, dataloader, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_metric = 0.0
    index=[]
    lr=[]
    losses=[]
    ind = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        more_toxic_ids = data['more_toxic_ids'].cuda()
        more_toxic_mask = data['more_toxic_mask'].cuda()
        less_toxic_ids = data['less_toxic_ids'].cuda()
        less_toxic_mask = data['less_toxic_mask'].cuda()
        targets = data['target'].cuda()
        
        batch_size = args.batch_size

        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        
        loss = nn.MarginRankingLoss(margin=args.margin)(more_toxic_outputs, less_toxic_outputs, targets)
        loss = loss / args.accumulation_step
        loss.backward()
        
        losses.append(loss.item())
        if (step + 1) % args.accumulation_step == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        score = metric(more_toxic_outputs, less_toxic_outputs)
        running_metric += score * batch_size

        epoch_loss = running_loss / dataset_size
        epoch_score = running_metric / dataset_size
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        score=epoch_score)
        ind=ind+1
        index.append(ind)
        lr.append(optimizer.param_groups[0]['lr'])
        
    gc.collect()
    
    return epoch_loss, index, lr, losses



def valid_one_epoch(model, dataloader, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_metric = 0.0
    losses=[]
    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:        
            more_toxic_ids = data['more_toxic_ids'].cuda()
            more_toxic_mask = data['more_toxic_mask'].cuda()
            less_toxic_ids = data['less_toxic_ids'].cuda()
            less_toxic_mask = data['less_toxic_mask'].cuda()
            targets = data['target'].cuda()

            batch_size = 2*args.batch_size

            more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
            less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)

            loss = nn.MarginRankingLoss(margin=args.margin)(more_toxic_outputs, less_toxic_outputs, targets)
            losses.append(loss.item())
            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            score = metric(more_toxic_outputs, less_toxic_outputs)
            running_metric += score * batch_size

            epoch_loss = running_loss / dataset_size
            epoch_score = running_metric / dataset_size

            bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                            score=epoch_score)   

        gc.collect()

        return epoch_loss, losses, epoch_score

def get_loaders(args, fold, df,tokenizer):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = ToxicDataset(df_train, tokenizer=tokenizer, max_length=args.max_length)
    valid_dataset = ToxicDataset(df_valid, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=2*args.batch_size, 
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader, len(train_dataset)

def run(args, model, optimizer, scheduler, num_epochs, fold):
    
    if torch.cuda.is_available():
        print("Model pushed to GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()

    best_epoch_loss = np.inf
    best_epoch_score = 0
    patience_counter = 0
    indexes=[]
    idx=0
    lrs=[]
    train_losses=[]
    valid_losses=[]
    epoch_train_losses=[]
    epoch_valid_losses=[]
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss, index, lr, train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           epoch=epoch)
        epoch_train_losses.append(train_epoch_loss)
        idx = idx+len(index)
        

        lrs.append(lr)
        train_losses.append(train_loss)
        
        
        val_epoch_loss, val_loss, val_score = valid_one_epoch(model, valid_loader,
                                         epoch=epoch)
        epoch_valid_losses.append(val_epoch_loss)
        valid_losses.append(val_loss)
        
        if val_score >= best_epoch_score:
            
            print(f"Validation Score Improved ({best_epoch_score} ---> {val_score})")
            best_epoch_score = val_score

            PATH = f"model_fold_{fold}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")
            
        else:
            patience_counter+=1
            print('-'*50)
            print(f'Early stopping counter {patience_counter} of {args.patience}')
            print('-'*50)
            if patience_counter == args.patience:
                print('*'*20,'Early Stopping','*'*20)
                break
            
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    
    
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_epoch_score))
    
    # load best model weights

    
    return model

def fetch_scheduler(args, optimizer, num_training_steps):
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=500, 
                                                   eta_min=1e-6)
#     elif args.schedular == 'CosineAnnealingWarmRestarts':
#         scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
#                                                              eta_min=CONFIG['min_lr'])

    elif args.scheduler == 'LinearWarmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=args.warmup_steps, 
            num_training_steps=num_training_steps
        )
    elif args.schedular == None:
        return None
        
    return scheduler




if __name__ == '__main__':

    df = pd.read_csv('5folds.csv')

    for fold in range(0,5):
        print('-'*50)
        print(f"Fold: {fold}")
        print('-'*50)
        
        args = Config()
        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_loader, valid_loader, train_dataset_len = get_loaders(args, fold, df, tokenizer)
        
        model = ToxicModel(args.model_name, args)
        model = model.cuda()
        num_training_steps = (train_dataset_len / args.batch_size * args.epochs)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = fetch_scheduler(args, optimizer, num_training_steps)
        
        model = run(args, model, optimizer, scheduler,
                                    num_epochs=args.epochs,
                                    fold=fold)
        
        del model, train_loader, valid_loader, train_dataset_len
        _ = gc.collect()
        print('-'*100)
        print()


