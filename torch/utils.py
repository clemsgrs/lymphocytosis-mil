import json
import torch
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from samplers import TopKSampler


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def open_config_file(filepath):
    with open(filepath) as jsonfile:
        pdict = json.load(jsonfile)
        params = AttrDict(pdict)
    return params


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run_inference(epoch, model, inference_loader, df, criterion, topk_processor, params, threshold=0.5):

    model.eval()
    epoch_loss = 0
    instance_indices = [-1] * len(inference_loader.dataset)
    probs = torch.FloatTensor(len(inference_loader.dataset))

    with tqdm(inference_loader,
              desc=(f'Inference - Epoch: {epoch}'),
              unit=' img',
              ncols=80,
              unit_scale=params.batch_size) as t:

        with torch.no_grad():
            
            for i, batch in enumerate(t):

                index, image, lymph_count, label = batch
                index, image, lymph_count, label = index.cuda(), image.cuda(), lymph_count.cuda(), label.cuda()
                output = model(image, lymph_count)
                loss = criterion(output, label.float())
                prob = output.detach()

                probs[i*params.batch_size:i*params.batch_size+index.size(0)] = prob.clone()
                instance_indices[i*params.batch_size:i*params.batch_size+index.size(0)] = list(index)
                
                epoch_loss += loss.item()
        
        df.loc[instance_indices, 'inference_prob'] = probs.cpu()
        topk_indices = topk_processor(
            df,
            prob_col_name='inference_prob',
            group='id'
        )
        patient_ids, probs, preds, labels = self.topk_processor.aggregate(
            df,
            topk_indices,
            prob_col_name='inference_prob',
            group='id', 
            threshold=0.5,
        )

        metrics = get_metrics(probs, preds, labels)
        metrics = {m: v / len(inference_loader) for m,v in metrics.items()}
        avg_loss = epoch_loss / len(inference_loader)
        train_sampler = TopKSampler(topk_indices)

        return avg_loss, metrics, train_sampler


def run_training(epoch, model, train_loader, df, optimizer, criterion, topk_processor, params, threshold=0.5):

    model.train()
    epoch_loss = 0
    instance_indices = [-1] * len(train_loader.dataset)
    probs = torch.FloatTensor(len(train_loader.dataset))

    with tqdm(train_loader,
              desc=(f'Train - Epoch: {epoch}'),
              unit=' img',
              ncols=80,
              unit_scale=params.batch_size) as t:

        for i, batch in enumerate(t):

            optimizer.zero_grad()
            index, image, lymph_count, label = batch
            index, image, lymph_count, label = index.cuda(), image.cuda(), lymph_count.cuda(), label.cuda()
            output = model(image, lymph_count)
            
            prob = output.detach()
            probs[i*params.batch_size:i*params.batch_size+index.size(0)] = prob.clone()
            instance_indices[i*params.batch_size:i*params.batch_size+index.size(0)] = list(index)

            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        df.loc[instance_indices, 'training_prob'] = probs.cpu()
        patient_ids, probs, preds, labels = topk_processor.aggregate(
            df,
            instance_indices,
            prob_col_name='training_prob',
            group='id', 
            threshold=0.5,
        )

        metrics = get_metrics(probs, preds, labels)
        metrics = {m: v / len(train_loader) for m,v in metrics.items()}
        avg_loss = epoch_loss / len(train_loader)
        
        return avg_loss, metrics


def run_validation(epoch, model, val_loader, df, criterion, topk_processor, params, threshold=0.5):

    model.eval()
    epoch_loss = 0
    instance_indices = [-1] * len(train_loader.dataset)
    probs = torch.FloatTensor(len(train_loader.dataset))

    with tqdm(val_loader,
              desc=(f'Train - Epoch: {epoch}'),
              unit=' img',
              ncols=80,
              unit_scale=params.batch_size) as t:

        with torch.no_grad():
            
            for i, batch in enumerate(t):

                index, image, lymph_count, label = batch
                index, image, lymph_count, label = index.cuda(), image.cuda(), lymph_count.cuda(), label.cuda()
                output = model(image, lymph_count)
                prob = output.detach()
                loss = criterion(output, label.float())

                probs[i*params.batch_size:i*params.batch_size+index.size(0)] = prob.clone()
                instance_indices[i*params.batch_size:i*params.batch_size+index.size(0)] = list(index)
                
                epoch_loss += loss.item()
        
        df.loc[instance_indices, 'validation_prob'] = probs.cpu()
        topk_indices = topk_processor(
            df,
            prob_col_name='validation_prob',
            group='id'
        )
        patient_ids, probs, preds, labels = topk_processor.aggregate(
            df,
            topk_indices,
            prob_col_name='validation_prob',
            group='id', 
            threshold=0.5,
        )

        metrics = get_metrics(probs, preds, labels)
        metrics = {m: v / len(val_loader) for m,v in metrics.items()}
        avg_loss = epoch_loss / len(val_loader)
        
        return avg_loss, metrics


def test_model(model, test_loader, params, threshold=0.5):

    model.eval()
    preds_dict = {}

    with tqdm(test_loader,
             desc=(f'Test: '),
             unit=' patient',
             ncols=80,
             unit_scale=params.test_batch_size) as t:

        with torch.no_grad():

            for i, (signal, sample_index, subject_index) in enumerate(t):

                signal = signal.type(torch.FloatTensor)
                signal = signal.cuda()
                preds = model(signal).unsqueeze(0)
                preds = preds.type(torch.FloatTensor).cpu()
                sample_index = sample_index.item()
                
                preds_dict[int(sample_index)] = [int(x>threshold) for x in preds.tolist()]

    preds_df = format_prediction_to_submission_canvas(preds_dict)
    return preds_df


def plot_curves(train_losses, train_accuracies, validation_losses, validation_accuracies, params):

    x = range(params.nepochs)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x, train_losses, label='train', color='#6F1BDA')
    axs[0].plot(x, validation_losses, label='val', color='#DA1BC6')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].set_title('Loss')
    axs[0].legend()
    axs[1].plot(x, train_accuracies, label='train', color='#6F1BDA')
    axs[1].plot(x, validation_accuracies, label='val', color='#DA1BC6')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('acc')
    axs[1].set_title('Accuracy')
    axs[1].legend()

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig('curves.pdf')
    plt.show()


def get_metrics(probs, preds, labels):
    labels = labels.type(torch.IntTensor)
    probs, preds, labels = probs.numpy(), preds.numpy(), labels.numpy()
    
    acc = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    
    metrics = {'acc': acc, 'balanced_acc': balanced_acc, 'auc': auc, 'precision': precision, 'recall': recall}
    return metrics
