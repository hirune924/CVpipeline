#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import yaml
import json
import datetime

from absl import flags, app
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

try:
    from torchsummary import summary as tsummary
except ImportError:
    print('[WARNING] {} module is not installed'.format('torchsummary'))

try:
    from apex import amp
except ImportError:
    print('[WARNING] {} module is not installed'.format('apex'))

from dataset.dataset import load_data
from dataset.dalidataset import DALIDataLoader
from model.model import get_model_from_name
from optimizer.optimizer import get_optimizer_from_name
from loss.loss import get_loss_from_name

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

flags.DEFINE_string('config_path', default='configs/base.yml', help='path to config file', short_name='c')
FLAGS = flags.FLAGS


def train_loop(model, loader, criterion, optimizer, amp=False):
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
        if amp:
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

def main(argv=None):
    # Load Config
    with open(FLAGS.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    print('Configs overview:')
    print(json.dumps(config, indent=2))

    # define SummaryWriter for tensorboard
    writer = SummaryWriter(log_dir=config['tensorboard']['log_dir'] + '-' + str(datetime.datetime.now()))
    writer.add_text('config', json.dumps(config, indent=2))

    # define dataloader
    dataset_options = config['dataset']
    dataloader_options = config['dataloader']
    if dataset_options['mode'] == 'dali':
       train_data_loader = DALIDataLoader(csv_path = dataset_options['train_csv_path'],
                                          data_path = dataset_options['train_image_path'],
                                          batch_size=dataloader_options['batch_size'], valid=False,
                                          nfold=dataset_options['validation']['nfold']) 
       valid_data_loader = DALIDataLoader(csv_path = dataset_options['train_csv_path'],
                                          data_path = dataset_options['train_image_path'],
                                          batch_size=dataloader_options['batch_size'], valid=True,
                                           nfold=dataset_options['validation']['nfold']) 
    else:
        train_data_loader, valid_data_loader = load_data(data_path=dataset_options['train_image_path'],
                                                               csv_path=dataset_options['train_csv_path'],
                                                               batch_size=dataloader_options['batch_size'],
                                                               valid_mode=dataset_options['validation']['mode'],
                                                               nfold=dataset_options['validation']['nfold'],
                                                               mode=dataset_options['mode'],
                                                               lmdb_path=dataset_options['lmdb_path'])
    
    # define model
    model_options = config['model']
    model = get_model_from_name(model_name=model_options['model_name'],
                                image_size=model_options['image_size'],
                                num_classes=model_options['num_classes'],
                                pretrained=model_options['pretrained'])
    # for tensorboard and torchsummary
    image_size = model_options['image_size']
    writer.add_graph(model, torch.zeros(image_size).unsqueeze(dim=0))
    model = model.to(DEVICE)
    if 'torchsummary' in sys.modules:
        tsummary(model, tuple(image_size))

    # define optimizer
    opt_options = config['optimizer']
    opt_params = opt_options['opt_params']
    optimizer = get_optimizer_from_name(opt_name=opt_options['opt_name'], 
                                        model=model, 
                                        **opt_params)

    # define loss
    loss_options = config['loss']
    loss_params = loss_options['loss_params']
    loss_fn = get_loss_from_name(loss_name=loss_options['loss_name'],
                                   **loss_params)

    # Mixed Precision
    amp_options = config['amp']
    use_amp = 'amp' in sys.modules and amp_options['use_amp']
    if use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_options['opt_level'], num_losses=1)
        if model_options['dataparallel']:
            model = nn.DataParallel(model)
    elif model_options['dataparallel']:
        model = nn.DataParallel(model)



    for e in range(1, config['train']['epoch'] + 1):
        train_loss, train_acc = train_loop(model, train_data_loader, loss_fn, optimizer, amp=use_amp)
        writer.add_scalar('Train/Loss', train_loss, e)
        writer.add_scalar('Train/Accuracy', train_acc, e)

        valid_loss, valid_acc = valid_loop(model, valid_data_loader, loss_fn)
        writer.add_scalar('Validation/Loss', valid_loss, e)
        writer.add_scalar('Validation/Accuracy', valid_acc, e)

        print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}, Valid Loss: {:.4f}, Valid Accuracy: {:.2f}'.format(e, train_loss, train_acc, valid_loss, valid_acc))
        writer.add_scalars('Summary/Loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, e)
        writer.add_scalars('Summary/Accuracy', {'train_acc': train_acc, 'valid_acc': valid_acc}, e)

    writer.flush()
    writer.close()

if __name__ == '__main__':
    app.run(main)

