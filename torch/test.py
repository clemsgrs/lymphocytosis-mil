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

from models import create_model
from dataset import TestDataModule
from utils import *

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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# load best weights from training (based on params.tracking value)
model = create_model(params)
model.load_state_dict(torch.load('best_model.pt'))
model = model.cuda()
model.eval()

test_predictions_df = run_test(model, test_loader, params, threshold=threshold)
test_predictions_df.to_csv(f'test_predictions.csv', index=False)
print('done')