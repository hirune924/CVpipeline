#!/usr/bin/env python
# -*- coding: utf-8 -*-
import yaml
import json
from absl import flags, app
from dataset.dataset import load_train_data

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
    


if __name__ == '__main__':
    app.run(main)

