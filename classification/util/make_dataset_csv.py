#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
from absl import flags, app

flags.DEFINE_string('csv_path', default=None, help='path to csv file', short_name='csv')
flags.DEFINE_string('trans_json', default=None, help='save json info for transform', short_name='tjson')
flags.DEFINE_string('trans_csv', default=None, help='save transformed csv', short_name='tcsv')
flags.DEFINE_string('add_img_format', default=None, help='add image format', short_name='addf')

FLAGS = flags.FLAGS

def main(argv=None):

    # Load csv
    df = pd.read_csv(FLAGS.csv_path)

    # Rename columns 
    cols = df.columns
    df = df.rename(columns={cols[0]: 'image', cols[1]: 'label'}).loc[:,['image','label']]

    # Make transform dict
    class_names = df['label'].drop_duplicates()
    print('num classes: {}'.format(len(class_names)))

    codes = range(len(class_names))

    name_to_code = dict(zip(class_names.sort_values(), codes))
    code_to_name = dict(zip(codes, class_names.sort_values()))

    print('transform information:\n {}'.format(json.dumps(code_to_name, indent=2)))

    # Save transform json if you need
    if FLAGS.trans_json is not None:
        with open(FLAGS.trans_json, 'w') as f:
            json.dump({'code_to_name': code_to_name, 'name_to_code': name_to_code}, f, indent=2)

    # Transform label
    df['label'] = [name_to_code[x] for x in df['label']]

    # Add image file format if you need
    if FLAGS.add_img_format is not None:
        df['image'] = df['image'] + '.' + FLAGS.add_img_format

    # Save transformed csv if you need
    if FLAGS.trans_csv is not None:
        df.to_csv(FLAGS.trans_csv, index=False)

    print('transformed:\n {}'.format(df))



if __name__ == '__main__':
    app.run(main)
