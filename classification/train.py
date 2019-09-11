#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import yaml
import json
from absl import flags, app
from dataset.dataset import load_train_data
from model.model import get_model_from_name
from optimizer.optimizer import get_optimizer_from_name
from loss.loss import get_loss_from_name

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

flags.DEFINE_string('config_path', default='configs/base.yml', help='path to config file', short_name='c')
FLAGS = flags.FLAGS

def main(argv=None):
    # Load Config
    with open(FLAGS.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    print('Configs overview:')
    print(json.dumps(config, indent=2))

    train_data_loader = load_train_data(data_path=config['dataset']['train_image_path'],
                                        csv_path=config['dataset']['train_csv_path'],
                                        batch_size=config['dataloader']['batch_size'])
    
    model = get_model_from_name(model_name=config['model']['model_name'],
                                image_size=config['model']['image_size'],
                                num_classes=config['model']['num_classes'],
                                pretrained=True)
    model = model.to(DEVICE)

    optimizer = get_optimizer_from_name(opt_name=config['optimizer']['opt_name'], 
                                        model=model, 
                                        **config['optimizer']['opt_params'])

    loss = get_loss_from_name(loss_name=config['loss']['loss_name'],
                              **config['loss']['loss_params'])

if __name__ == '__main__':
    app.run(main)

