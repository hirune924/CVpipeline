#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
import numpy as np
try:
    from apex import amp
except ImportError:
    print('[WARNING] {} module is not installed'.format('apex'))


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_loop(model, loader, criterion, optimizer, metric_dict, use_amp=False):
    model.train()
    y_pred = []
    y_true = []
    bar = tqdm(total=len(loader), leave=False)
    total_loss, total_acc, total_num = 0, 0, 0
    for idx, feed in enumerate(loader):
        # Prepare data
        inputs, labels = feed
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        # Foward
        outputs = model(inputs)
        # Calcurate Loss
        loss = criterion(outputs, labels)
        # initialize gradient
        optimizer.zero_grad()
        # Backward
        if use_amp:
            with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # Update Params
        optimizer.step()

        ## for metric
        y_pred.extend(outputs.data.cpu().numpy().tolist())
        y_true.extend(labels.data.cpu().numpy().tolist()) 
        ## Accuracy
        pred = outputs.data.max(1, keepdim=True)[1]
        acc = pred.eq(labels.data.view_as(pred)).sum()
        ## Calcurate Score
        total_loss += loss.item() * labels.size(0)
        total_acc += acc.item()
        total_num += labels.size(0)
        # Update bar
        bar.set_description("Loss: {:.4f}, Accuracy: {:.2f}".format(
            total_loss / total_num, total_acc / total_num * 100), refresh=True)
        bar.update()
    bar.close()
    scores = calc_metric(np.asarray(y_true), np.asarray(y_pred), metric_dict, {'loss': total_loss / total_num})
    #return total_loss / total_num, total_acc / total_num * 100
    return scores



def valid_loop(model, loader, criterion, metric_dict):
    model.eval()
    y_pred = []
    y_true = []
    total_loss, total_acc, total_num = 0, 0, 0
    bar = tqdm(loader, total=len(loader), leave=False)
    for i, feed in enumerate(loader):
        with torch.no_grad():
            # Prepare data
            inputs, labels = feed
            inputs = inputs.cuda()
            labels = labels.cuda()
            # Foward
            outputs = model(inputs)
            # Calcurate Loss
            loss = criterion(outputs, labels)
            ## for metric
            y_pred.extend(outputs.data.cpu().numpy().tolist())
            y_true.extend(labels.data.cpu().numpy().tolist()) 
            ## Accuracy
            pred = outputs.data.max(1, keepdim=True)[1]
            acc = pred.eq(labels.data.view_as(pred)).sum()
            ## Calcurate Score
            total_loss += loss.item() * labels.size(0)
            total_acc += acc.item()
            total_num += labels.size(0)
            # Update bar
            bar.set_description("Loss: {:.4f}, Accuracy: {:.2f}".format(
                total_loss / total_num, total_acc / total_num * 100), refresh=True)
            bar.update()
    bar.close()
    scores = calc_metric(np.asarray(y_true), np.asarray(y_pred), metric_dict, {'loss': total_loss / total_num})
    #return total_loss / total_num, total_acc / total_num * 100
    return scores

def save_params(keys=None, targets=None, save_path=None):
    checkpoint = {}
    if len(keys) != len(targets):
        print('[Warning] Length of keys and target is different!!')
    for k, t in zip(keys, targets):
        checkpoint[k] = t.state_dict()
    torch.save(checkpoint, save_path)

def calc_metric(y_true, y_pred, metrics, append):
    scores = {}
    for name, cls in metrics.items():
        scores[name] = cls.calc_score(y_true, y_pred)
    for name, s in append.items():
        scores[name] = s
    return scores
    
def print_save_scores(score, epoch, prefix, writer, display=True):
    message = 'Epoch: {}, '.format(epoch)
    'Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}, Valid Loss: {:.4f}, Valid Accuracy: {:.2f}'
    for name, s in score.items():
        message = message + prefix + ' ' + name + ': {:.4f}, '.format(s)
        writer.add_scalar(prefix + '/' + name, s, epoch)
    if display:
        print(message[:-1])

def print_save_scores_summary(train_score, valid_score, epoch, writer, display=True):
    message = 'Epoch: {}, '.format(epoch)
    'Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}, Valid Loss: {:.4f}, Valid Accuracy: {:.2f}'
    for name, s in train_score.items():
        message = message + 'Train ' + name + ': {:.4f}, '.format(s)
    for name, s in valid_score.items():
        message = message + 'Valid ' + name + ': {:.4f}, '.format(s)
    for name, s in valid_score.items():
        if name in train_score.keys():
            writer.add_scalars('Summary/{}'.format(name), {'train_{}'.format(name): train_score[name], 'valid_{}'.format(name): s}, epoch)
    if display:
        print(message[:-2])
