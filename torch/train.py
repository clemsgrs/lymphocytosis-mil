import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

data_module = LymphoDataModule(params.data_dir, val_size=params.val_size, seed=params.seed)
data_module.setup()
train_dataset, val_dataset = data_module.train_dataset, data_module.val_dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)

topk_processor = TopKProcessor(topk=params.topk, aggregation=params.aggregation)
train_df = train_dataset.df
val_df = val_dataset.df

### TRAINING

model = se_resnet50()
model.fc = nn.Linear(2048, 1)
optimizer = optim.Adam(model.parameters(), lr=params.lr)
if params.lr_scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step, gamma=0.1)
model = model.cuda()

criterion = nn.BCELoss()
criterion = criterion.cuda()

best_val_loss = float('inf')
best_val_acc = 0.0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(params.nepochs):

    start_time = time.time()
    inference_loss, inference_metrics, train_sampler = run_inference(
        epoch+1,
        model,
        train_loader,
        train_df,
        criterion,
        topk_processor, 
        params
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=params.batch_size,
        sampler=self.train_sampler,
        shuffle=False        
    )

    train_loss, train_metric = run_training(epoch+1, model, train_loader, train_df, optimizer, criterion, topk_processor, params)
    train_losses.append(train_loss)
    train_metrics.append(train_metric)

    if epoch % params.eval_every == 0:
        
        val_loss, val_metric = run_validation(epoch, model, val_loader, val_df, criterion, topk_processor, params)
        val_losses.append(val_loss)
        val_accuracies.append(val_metric)

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
    print(f'Train loss: {np.round(train_loss,6)} \t Train acc: {np.round(train_acc,4)}')
    print(f'Val loss: {np.round(val_loss,6)} \t Val acc: {np.round(val_acc,4)}\n')