import json
import torch
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
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


def run_inference(epoch, model, inference_dataset, criterion, topk_processor, params, threshold=0.5):

    epoch_loss = 0
    instance_indices = []
    probs = []

    inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=params.batch_size, shuffle=False)

    with tqdm(inference_loader,
              desc=(f'Inference - Epoch {epoch}'),
              unit=' img',
              ncols=80,
              unit_scale=params.batch_size) as t:

        with torch.no_grad():
            
            for i, batch in enumerate(t):

                index, image, lymph_count, label = batch
                image, lymph_count, label = image.cuda(), lymph_count.cuda(), label.cuda()
                logits = model(image)
                loss = criterion(logits, label.float())
                
                prob = torch.sigmoid(logits)
                probs.extend(prob[:,0].clone().tolist())
                instance_indices.extend(list(index))
                
                epoch_loss += loss.item()
        
        inference_dataset.df.loc[instance_indices, 'inference_prob'] = probs
        topk_indices = topk_processor(
            inference_dataset.df,
            prob_col_name='inference_prob',
            group='id'
        )
        patient_ids, probs, preds, labels = topk_processor.aggregate(
            inference_dataset.df,
            topk_indices,
            prob_col_name='inference_prob',
            group='id', 
            threshold=0.5,
        )

        # metrics = get_metrics(probs, preds, labels)
        best_balanced_acc, best_threshold = get_balanced_accuracy(probs, labels, thresholds=np.arange(0.0, 1, 0.01))
        avg_loss = epoch_loss / len(inference_loader)
        train_sampler = TopKSampler(topk_indices)

        return avg_loss, best_balanced_acc, best_threshold, train_sampler


def run_training(epoch, model, train_dataset, train_sampler, optimizer, criterion, topk_processor, params, threshold=0.5):

    model.train()
    epoch_loss = 0
    instance_indices = []
    probs = []

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=params.batch_size,
        sampler=train_sampler,
        shuffle=False        
    )

    with tqdm(train_loader,
              desc=(f'Train - Epoch {epoch}'),
              unit=' img',
              ncols=80,
              unit_scale=params.batch_size) as t:

        for i, batch in enumerate(t):

            optimizer.zero_grad()
            index, image, lymph_count, label = batch
            image, lymph_count, label = image.cuda(), lymph_count.cuda(), label.cuda()
            logits = model(image)
            loss = criterion(logits, label.float())
            loss.backward()
            optimizer.step()

            prob = torch.sigmoid(logits)
            probs.extend(prob[:,0].clone().tolist())
            instance_indices.extend(list(index))            
            
            epoch_loss += loss.item()
        
        train_dataset.df.loc[instance_indices, 'training_prob'] = probs
        patient_ids, probs, preds, labels = topk_processor.aggregate(
            train_dataset.df,
            instance_indices,
            prob_col_name='training_prob',
            group='id', 
            threshold=0.5,
        )

        # metrics = get_metrics(probs, preds, labels)
        best_balanced_acc, best_threshold = get_balanced_accuracy(probs, labels, thresholds=np.arange(0.0, 1, 0.01), plot=True)
        avg_loss = epoch_loss / len(train_loader)
        
        return avg_loss, best_balanced_acc, best_threshold


def run_validation(epoch, model, val_dataset, criterion, topk_processor, params, threshold=0.5):

    model.eval()
    epoch_loss = 0
    instance_indices = []
    probs = []

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)

    with tqdm(val_loader,
              desc=(f'Validation - Epoch {epoch}'),
              unit=' img',
              ncols=80,
              unit_scale=params.batch_size) as t:

        with torch.no_grad():
            
            for i, batch in enumerate(t):

                index, image, lymph_count, label = batch
                image, lymph_count, label = image.cuda(), lymph_count.cuda(), label.cuda()
                logits = model(image)
                loss = criterion(logits, label.float())
                
                prob = torch.sigmoid(logits)
                probs.extend(prob[:,0].clone().tolist())
                instance_indices.extend(list(index))
                
                epoch_loss += loss.item()
        
        val_dataset.df.loc[instance_indices, 'validation_prob'] = probs
        topk_indices = topk_processor(
            val_dataset.df,
            prob_col_name='validation_prob',
            group='id'
        )
        patient_ids, probs, preds, labels = topk_processor.aggregate(
            val_dataset.df,
            topk_indices,
            prob_col_name='validation_prob',
            group='id', 
            threshold=0.5,
        )

        # metrics = get_metrics(probs, preds, labels)
        best_balanced_acc, best_threshold = get_balanced_accuracy(probs, labels, thresholds=np.arange(0.0, 1, 0.01), plot=True)
        avg_loss = epoch_loss / len(val_loader)
        
        return avg_loss, best_balanced_acc, best_threshold


def run_test(model, test_dataset, topk_processor, params, threshold=0.5):

    model.eval()
    instance_indices = []
    probs = []
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)    

    with tqdm(test_loader,
             desc=(f'Test '),
             unit=' patient',
             ncols=80,
             unit_scale=params.test_batch_size) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                index, image, lymph_count, label = batch
                image, lymph_count, label = image.cuda(), lymph_count.cuda(), label.cuda()
                logits = model(image)                
                prob = torch.sigmoid(logits)
                probs.extend(prob[:,0].clone().tolist())
                instance_indices.extend(list(index))
                
                epoch_loss += loss.item()

    test_dataset.df.loc[instance_indices, 'prob'] = probs
    topk_indices = topk_processor(
        test_dataset.df,
        prob_col_name='prob',
        group='id'
    )
    patient_ids, probs, preds, labels = topk_processor.aggregate(
        test_dataset.df,
        topk_indices,
        prob_col_name='prob',
        group='id', 
        threshold=threshold,
    )
    
    preds_dict = {'Id': list(patient_ids), 'Predicted': list(preds.numpy())}
    test_predictions_df = pd.DataFrame.from_dict(test_preds_dict)
    
    return test_predictions_df


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


def get_metrics(probs, labels):
    probs, labels = probs.numpy(), labels.numpy()
    preds = probs > 0.5
    acc = metrics.accuracy_score(labels, preds)
    balanced_acc = metrics.balanced_accuracy_score(labels, preds)
    auc = metrics.roc_auc_score(labels, probs)
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds)
    
    metrics_dict = {'acc': acc, 'balanced_acc': balanced_acc, 'auc': auc, 'precision': precision, 'recall': recall}
    return metrics_dict

def get_balanced_accuracy(probs, labels, thresholds=[0.5], plot=False):
    probs, labels = probs.numpy(), labels.numpy()
    accs = []
    for threshold in thresholds:
        preds = (probs > threshold).astype('int')
        balanced_acc = metrics.balanced_accuracy_score(labels, preds)
        accs.append(balanced_acc)
    
    best_balanced_acc = np.amax(accs)
    best_threshold = thresholds[np.argmax(accs)]

    if plot:
        plt.figure(figsize=(10,7))
        plt.plot(thresholds, accs, color='#506AA5')
        plt.axvline(x=best_threshold, color='#9950A5', linestyle=':')
        plt.axhline(y=best_balanced_acc, color='#9950A5', linestyle=':')
        plt.xlabel('threshold')
        plt.ylabel('balanced accuracy')
        plt.title('balanced accuracy as a function of threshold')
        plt.show()
    
    return best_balanced_acc, best_threshold

