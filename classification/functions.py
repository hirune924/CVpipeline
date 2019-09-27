#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
try:
    from apex import amp
except ImportError:
    print('[WARNING] {} module is not installed'.format('apex'))


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_loop(model, loader, criterion, optimizer, use_amp=False):
    model.train()
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
    return total_loss / total_num, total_acc / total_num * 100



def valid_loop(model, loader, criterion):
    model.eval()
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
    return total_loss / total_num, total_acc / total_num * 100

def save_params(keys=None, targets=None, save_path=None):
    checkpoint = {}
    if len(keys) != len(targets):
        print('[Warning] Length of keys and target is different!!')
    for k, t in zip(keys, targets):
        checkpoint[k] = t.state_dict()
    torch.save(checkpoint, save_path)
