#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import pickle
import cv2
from absl import flags, app
import lmdb

flags.DEFINE_string('csv_path', default=None, help='path to csv file', short_name='csv')
flags.DEFINE_string('img_dir_path', default=None, help='Image dir path', short_name='imdir')
flags.DEFINE_string('lmdb_name', default=None, help='LMDB name', short_name='lmdb')

FLAGS = flags.FLAGS

def main(argv=None):

    # Load csv
    df = pd.read_csv(FLAGS.csv_path)

    # create lmdb
    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(FLAGS.lmdb_name, map_size=map_size)

    with env.begin(write=True) as txn:
        for index, row in df.iterrows():
            print(index, row['image'])
            img_path = os.path.join(FLAGS.img_dir_path, row['image'])
            img = cv2.imread(img_path)
            txn.put(row['image'].encode('utf-8'), pickle.dumps(img))
        txn.put('length'.encode('utf-8'), str(len(df)).encode('utf-8'))

if __name__ == '__main__':
    app.run(main)
