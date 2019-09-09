#!/usr/bin/env python
# -*- coding: utf-8 -*-
import yaml
import json
from absl import flags, app

flags.DEFINE_string('config_path', default='configs/base.yml', help='path to config file', short_name='c')
FLAGS = flags.FLAGS

def main(argv=None):
    # Load Config
    with open(FLAGS.config_path, 'r') as f:
        config = yaml.load(f)

    print('Configs overview:')
    print(json.dumps(config, indent=2))


    


if __name__ == '__main__':
    app.run(main)

