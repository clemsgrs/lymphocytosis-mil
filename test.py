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

from dataset import TestDataModule
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


### TESTING

data_module = TestDataModule(params)
data_module.setup()
test_dataset = data_module.test_dataset

topk_processor = TopKProcessor(topk=params.topk_agg, aggregation=params.aggregation)

# load best weights from training (based on params.tracking value)
# model = create_model(params)
model = timm.create_model('resnet18', pretrained=False)
model.fc = nn.Linear(512, 1)
model.load_state_dict(torch.load('best_model.pt'))
model = model.cuda()
model.eval()

test_predictions_df = run_test(model, test_dataset, topk_processor, params, threshold=threshold)
test_predictions_df.to_csv(f'test_predictions.csv', index=False)
print('done')