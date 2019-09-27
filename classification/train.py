#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
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

from functions import train_loop, valid_loop, save_params
from dataset.dataset import load_data
from dataset.dalidataset import DALIDataLoader
from model.model import get_model_from_name
from optimizer.optimizer import get_optimizer_from_name
from loss.loss import get_loss_from_name

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATETIME = datetime.datetime.now()
torch.backends.cudnn.benchmark = True

flags.DEFINE_string('config_path', default='configs/base.yml', help='path to config file', short_name='c')
FLAGS = flags.FLAGS


def main(argv=None):
    # Load Config
    with open(FLAGS.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    print('Configs overview:')
    print(json.dumps(config, indent=2))

    # define SummaryWriter for tensorboard
    log_dir = config['log']['log_dir'] + ';' + str(DATETIME).replace(' ', ';')
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
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
                                                               train_trans_list=dataloader_options['data_augment']['train_trans'],
                                                               valid_trans_list=dataloader_options['data_augment']['valid_trans'],
                                                               trans_mode=dataloader_options['data_augment']['mode'],
                                                               trans_lib=dataloader_options['data_augment']['lib'],
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
                                        target=opt_options['target'], 
                                        opt_params=opt_params,
                                        mode=opt_options['mode'],
                                        lib=opt_options['lib'])

    # define loss
    loss_options = config['loss']
    loss_params = loss_options['loss_params']
    loss_fn = get_loss_from_name(loss_name=loss_options['loss_name'],
                                 loss_params=loss_params,
                                 mode=loss_options['mode'],
                                 lib=loss_options['lib'])
    # Mixed Precision
    amp_options = config['amp']
    use_amp = ('apex' in sys.modules) and amp_options['use_amp']
    if use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_options['opt_level'], num_losses=1)
    if model_options['dataparallel']:
        model = nn.DataParallel(model)

    # for save params
    save_options = config['log']
    save_dir = os.path.join(log_dir, 'checkpoints')
    os.mkdir(save_dir)
    save_keys = ['model', 'optimizer']
    save_target = [model, optimizer]
    if use_amp:
        save_keys.append('amp')
        save_target.append(amp)
    
    best_val_acc = 0
    for e in range(1, config['train']['epoch'] + 1):
        # Training
        train_loss, train_acc = train_loop(model, train_data_loader, loss_fn, optimizer, use_amp=use_amp)
        writer.add_scalar('Train/Loss', train_loss, e)
        writer.add_scalar('Train/Accuracy', train_acc, e)

        # Validation
        valid_loss, valid_acc = valid_loop(model, valid_data_loader, loss_fn)
        writer.add_scalar('Validation/Loss', valid_loss, e)
        writer.add_scalar('Validation/Accuracy', valid_acc, e)

        # Summary
        print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}, Valid Loss: {:.4f}, Valid Accuracy: {:.2f}'.format(e, train_loss, train_acc, valid_loss, valid_acc))
        writer.add_scalars('Summary/Loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, e)
        writer.add_scalars('Summary/Accuracy', {'train_acc': train_acc, 'valid_acc': valid_acc}, e)
        
        if best_val_acc < valid_acc and save_options['save_best_val'] and save_options['save_skip_epoch'] < e:
            save_params(keys=save_keys, targets=save_target, save_path=os.path.join(save_dir, save_options['save_name'] + '-' +  str(e) + '-' + str(valid_acc) +'.pth'))
            best_val_acc = valid_acc
        elif e%save_options['save_interval'] == 0  and save_options['save_skip_epoch'] < e:
            save_params(keys=save_keys, targets=save_target, save_path=os.path.join(save_dir, save_options['save_name'] + '-' +  str(e) + '.pth'))

    if save_options['save_final']:
        save_params(keys=save_keys, targets=save_target, save_path=os.path.join(save_dir, save_options['save_name'] + '-final.pth'))
    writer.close()

if __name__ == '__main__':
    app.run(main)

