#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import yaml
import json

from absl import flags, app
from tqdm import tqdm

from dataset.dataset import load_train_data
from model.model import get_model_from_name
from optimizer.optimizer import get_optimizer_from_name
from loss.loss import get_loss_from_name

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

flags.DEFINE_string('config_path', default='configs/base.yml', help='path to config file', short_name='c')
FLAGS = flags.FLAGS


def train_loop(model, loader, criterion, optimizer):
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
        loss.backward()
        # Update Params
        optimizer.step()
        # Update bar
        ## Accuracy
        pred = outputs.data.max(1, keepdim=True)[1]
        acc = pred.eq(labels.data.view_as(pred)).sum()
        ## Calcurate Score
        total_loss += loss.item() * labels.size(0)
        total_acc += acc.item()
        total_num += labels.size(0)
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
            # Update bar
            ## Accuracy
            pred = outputs.data.max(1, keepdim=True)[1]
            acc = pred.eq(labels.data.view_as(pred)).sum()
            ## Calcurate Score
            total_loss += loss.item() * labels.size(0)
            total_acc += acc.item()
            total_num += labels.size(0)

            bar.set_description("Loss: {:.4f}, Accuracy: {:.2f}".format(
                total_loss / total_num, total_acc / total_num * 100), refresh=True)
            bar.update()
    bar.close()
    return total_loss / total_num, total_acc / total_num * 100

def main(argv=None):
    # Load Config
    with open(FLAGS.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    print('Configs overview:')
    print(json.dumps(config, indent=2))

    train_data_loader, valid_data_loader = load_train_data(data_path=config['dataset']['train_image_path'],
                                                           csv_path=config['dataset']['train_csv_path'],
                                                           batch_size=config['dataloader']['batch_size'],
                                                           valid_mode=config['dataset']['validation']['mode'],
                                                           nfold=config['dataset']['validation']['nfold'])
    
    model = get_model_from_name(model_name=config['model']['model_name'],
                                image_size=config['model']['image_size'],
                                num_classes=config['model']['num_classes'],
                                pretrained=True)
    model = model.to(DEVICE)

    opt_params = config['optimizer']['opt_params'] if 'opt_params' in config['optimizer'] else {}
    optimizer = get_optimizer_from_name(opt_name=config['optimizer']['opt_name'], 
                                        model=model, 
                                        **opt_params)

    loss_params = config['loss']['loss_params'] if 'loss_params' in config['loss'] else {}
    criterion = get_loss_from_name(loss_name=config['loss']['loss_name'],
                                   **loss_params)

    for e in range(config['train']['epoch']):
        train_loss, train_acc = train_loop(model, train_data_loader, criterion, optimizer)
        valid_loss, valid_acc = valid_loop(model, valid_data_loader, criterion)
        print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}, Valid Loss: {:.4f}, Valid Accuracy: {:.2f}'.format(e + 1, train_loss, train_acc, valid_loss, valid_acc))

if __name__ == '__main__':
    app.run(main)

