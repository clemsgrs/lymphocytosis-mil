import time
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
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

mean = torch.tensor([0.8183, 0.6977, 0.7034])
std = torch.tensor([0.1917, 0.2156, 0.0917])
if params.center_crop:
    t = transforms.Compose([
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
else:
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

data_module = LymphoDataModule(params.data_dir, val_size=params.val_size, pct=params.pct, seed=params.seed, transforms=t)
data_module.setup()
train_dataset, val_dataset = data_module.train_dataset, data_module.val_dataset
print()

topk_train_processor = TopKProcessor(topk=params.topk_train, aggregation=params.aggregation)
topk_val_processor = TopKProcessor(topk=params.topk_agg, aggregation=params.aggregation)

### TRAINING

model = timm.create_model(params.model, pretrained=True)
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
    inference_loss, inference_metric, inference_threshold , train_sampler = run_inference(
        epoch+1,
        model,
        train_dataset,
        criterion,
        topk_train_processor, 
        params
    )
    inference_losses.append(inference_loss)
    inference_metrics.append(inference_metric)

    train_loss, train_metric, train_threshold = run_training(
        epoch+1, 
        model, 
        train_dataset, 
        train_sampler,
        optimizer, 
        criterion, 
        topk_train_processor, 
        params
    )
    train_losses.append(train_loss)
    train_metrics.append(train_metric)
    # train_bacc = train_metric['balanced_acc']
    train_bacc = train_metric

    if epoch % params.eval_every == 0:
        
        val_loss, val_metric, val_threshold = run_validation(
            epoch+1, 
            model, 
            val_dataset, 
            criterion, 
            topk_val_processor, 
            params
        )
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        # val_bacc = val_metric['balanced_acc']
        val_bacc = val_metric

        if params.tracking == 'val_loss':
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')

        elif params.tracking == 'val_acc':
            if val_bacc > best_val_acc:
                best_val_acc = val_bacc
                torch.save(model.state_dict(), 'best_model.pt')

    if params.lr_scheduler:
        scheduler.step()

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'End of epoch {epoch+1} / {params.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s')
    print(f'Train loss: {train_loss:.5f} \t Train acc: {train_bacc:.4f} (threshold={train_threshold:.2f})')
    print(f'Val loss: {val_loss:.5f} \t Val acc: {val_bacc:.4f} (threshold={val_threshold:.2f})\n')