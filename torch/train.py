import time
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from models import se_resnet50
from dataset import LymphoDataModule
from utils import *
from processing import TopKProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config/default.json", metavar='N', help='config file')
args = parser.parse_args()
params = open_config_file(args.config)

print('------------ Options -------------')
for k, v in vars(params).items():
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')
print()

data_module = LymphoDataModule(params.data_dir, val_size=params.val_size, pct=params.pct, seed=params.seed)
data_module.setup()
train_dataset, val_dataset = data_module.train_dataset, data_module.val_dataset
inference_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)
print()

topk_processor = TopKProcessor(topk=params.topk, aggregation=params.aggregation)
train_df = train_dataset.df
val_df = val_dataset.df

### TRAINING

model = timm.create_model('resnet18', pretrained=True)
model.fc = nn.Linear(512, 1)
optimizer = optim.Adam(model.parameters(), lr=params.lr)
if params.lr_scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step, gamma=0.1)
model = model.cuda()

criterion = nn.BCEWithLogitsLoss()
criterion = criterion.cuda()

best_val_loss = float('inf')
best_val_acc = 0.0

inference_losses, train_losses, val_losses = [], [], []
inference_metrics, train_metrics, val_metrics = [], [], []

for epoch in range(params.nepochs):

    start_time = time.time()
    inference_loss, inference_metric, train_sampler = run_inference(
        epoch+1,
        model,
        inference_loader,
        train_df,
        criterion,
        topk_processor, 
        params
    )
    inference_losses.append(inference_loss)
    inference_metrics.append(inference_metric)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=params.batch_size,
        sampler=train_sampler,
        shuffle=False        
    )

    train_loss, train_metric = run_training(
        epoch+1, 
        model, 
        train_loader, 
        train_df, 
        optimizer, 
        criterion, 
        topk_processor, 
        params
    )
    train_losses.append(train_loss)
    train_metrics.append(train_metric)
    train_bacc = train_metric['balanced_acc']

    if epoch % params.eval_every == 0:
        
        val_loss, val_metric = run_validation(
            epoch+1, 
            model, 
            val_loader, 
            val_df, 
            criterion, 
            topk_processor, 
            params
        )
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        val_bacc = val_metric['balanced_acc']

        if params.tracking == 'val_loss':
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')

        elif params.tracking == 'val_acc':
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pt')

    if params.lr_scheduler:
        scheduler.step()

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'End of epoch {epoch+1} / {params.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s')
    print(f'Train loss: {train_loss:.5f} \t Train acc: {train_bacc:.4f}')
    print(f'Val loss: {val_loss:.5f} \t Val acc: {val_bacc:.4f}\n')