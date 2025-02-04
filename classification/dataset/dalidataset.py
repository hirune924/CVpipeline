#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import random
import cv2
import pickle
from PIL import Image

try:
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    print('[WARNING] {} module is not installed'.format('dali'))

class DALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, csv_path, data_path, valid=False, nfold=0):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id)
        self.data_path = data_path
        self.csv_file = csv_path
        self.valid = valid
        self.data = pd.read_csv(self.csv_file)

        if nfold > 0:
            self.data = self.data.sort_values(by=['image', 'label'])
            self.data = self.data.sample(frac=1, random_state=0).reset_index(drop=True)
            len_fold = int(len(self.data)/nfold)
            if valid:
                self.data = self.data[len_fold*(nfold-1):].reset_index(drop=True)
            else:
                self.data = self.data[:len_fold*(nfold-1)].reset_index(drop=True)
        self.data.to_csv('data/dali.txt', header=False, index=False, sep=' ')

        self.input = ops.FileReader(file_root=data_path, file_list='data/dali.txt')
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.random_resize = ops.Resize(device = "gpu",
                                 image_type = types.RGB,
                                 interp_type = types.INTERP_LINEAR)
        self.resize = ops.Resize(device = "gpu",
                                 image_type = types.RGB,
                                 interp_type = types.INTERP_LINEAR,
                                 resize_x=227., resize_y=227.)
        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            crop = (227, 227),
                                            image_type = types.RGB,
                                            mean = [128., 128., 128.],
                                            std = [1., 1., 1.])
        self.normalize = ops.NormalizePermute(device = "gpu",
                                              height=227, width=227,
                                              image_type = types.RGB,
                                              mean = [128., 128., 128.],
                                              std = [1., 1., 1.])
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.resize_rng = ops.Uniform(range = (256, 480))

    def define_graph(self):
        images, labels = self.input(name="Reader")
        images = self.decode(images)
        if self.valid:
            images = self.resize(images)
            output = self.normalize(images)
        else:
            images = self.random_resize(images, resize_shorter = self.resize_rng())
            output = self.cmn(images, crop_pos_x = self.uniform(),
                              crop_pos_y = self.uniform())
        return (output, labels)


class DALICustomIterator(DALIGenericIterator):
    def __init__(self, pipelines, output_map, size, auto_reset=False, fill_last_batch=True, dynamic_shape=False, last_batch_padded=False):
        super(DALICustomIterator, self).__init__(pipelines, output_map, size, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded)

    def __len__(self):
        return int(self._size / self.batch_size) + 1

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        feed = super().__next__()
        data = feed[0]['data']
        labels = feed[0]['label'].squeeze().long()
        return data, labels

def DALIDataLoader(csv_path, data_path, batch_size, valid=False, nfold=0):
    num_gpus = 1
    pipes = [DALIPipeline(batch_size=batch_size, num_threads=8, device_id=device_id, csv_path=csv_path, data_path=data_path, valid=valid, nfold=nfold) for device_id in range(num_gpus)]

    pipes[0].build()
    dali_iter = DALICustomIterator(pipes, ['data', 'label'], pipes[0].epoch_size("Reader"), auto_reset=True)
    return dali_iter

