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
    import lmdb
except ImportError:
    print('[WARNING] {} module is not installed'.format('lmdb'))
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
        self.resize = ops.Resize(device = "gpu",
                                 image_type = types.RGB,
                                 interp_type = types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            crop = (227, 227),
                                            image_type = types.RGB,
                                            mean = [128., 128., 128.],
                                            std = [1., 1., 1.])
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.resize_rng = ops.Uniform(range = (256, 480))

    def define_graph(self):
        images, labels = self.input(name="Reader")
        images = self.decode(images)
        images = self.resize(images, resize_shorter = self.resize_rng())
        output = self.cmn(images, crop_pos_x = self.uniform(),
                          crop_pos_y = self.uniform())
        return (output, labels)
class DALICustomIterator(DALIGenericIterator):
    def __init__(self, pipelines, output_map, size, auto_reset=False, fill_last_batch=True, dynamic_shape=False, last_batch_padded=False):
        super(DALICustomIterator, self).__init__(pipelines, output_map, size, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded)

    def __len__(self):
        return int(self._size / self.batch_size)

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter >= self._size:
            if self._auto_reset:
                self.reset()
            raise StopIteration
        feed = super().__next__()
        data = feed[0]['data']
        labels = feed[0]['label'].squeeze().long()
        return data, labels

def DALIDataLoader(csv_path, data_path, valid=False, nfold=0):
    num_gpus = 2
    pipes = [DALIPipeline(batch_size=32, num_threads=1, device_id = device_id, csv_path = csv_path, data_path = data_path, valid=False, nfold=5) for device_id in range(num_gpus)]

    pipes[0].build()
    dali_iter = DALICustomIterator(pipes, ['data', 'label'], pipes[0].epoch_size("Reader"))
    return dali_iter

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(worker_id)

class BasicDataset(Dataset):
    def __init__(self, csv_path, data_path, transform=None, valid=False, nfold=0):
        self.data_path = data_path
        self.csv_file = csv_path
        self.data = pd.read_csv(self.csv_file)
        self.transform = transform

        if nfold > 0:
            self.data = self.data.sort_values(by=['image', 'label'])
            self.data = self.data.sample(frac=1, random_state=0).reset_index(drop=True)
            len_fold = int(len(self.data)/nfold)
            if valid:
                self.data = self.data[len_fold*(nfold-1):].reset_index(drop=True)
            else:
                self.data = self.data[:len_fold*(nfold-1)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, self.data.loc[idx, 'image'])
        image = Image.open(img_name)

        label = self.data.loc[idx, 'label']

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class LMDBDataset(Dataset):
    def __init__(self, csv_path, lmdb_path, transform=None, valid=False, nfold=0):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(self.lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        self.csv_file = csv_path
        self.data = pd.read_csv(self.csv_file)
        self.transform = transform

        if nfold > 0:
            self.data = self.data.sort_values(by=['image', 'label'])
            self.data = self.data.sample(frac=1, random_state=0).reset_index(drop=True)
            len_fold = int(len(self.data)/nfold)
            if valid:
                self.data = self.data[len_fold*(nfold-1):].reset_index(drop=True)
            else:
                self.data = self.data[:len_fold*(nfold-1)].reset_index(drop=True)

        
        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_path)

        #with self.env.begin(write=False) as txn:
        #    self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return len(self.data)
        #return self.length

    def __getitem__(self, idx):
        key = self.data.loc[idx, 'image']
        with self.env.begin(write=False) as txn:
            cv2_row = pickle.loads(txn.get(key.encode('utf-8')))

        image = Image.fromarray(cv2.cvtColor(cv2_row, cv2.COLOR_BGR2RGB))

        label = self.data.loc[idx, 'label']
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_data(csv_path, data_path, batch_size, valid_mode='auto', nfold=0, mode='basic', lmdb_path=None):
    if mode == 'basic':
        dataset = BasicDataset
    elif mode == 'lmdb':
        dataset = LMDBDataset
        data_path = lmdb_path

   
    train_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    valid_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if valid_mode == 'auto':
        train_data_loader = DataLoader(
                dataset(csv_path, data_path,
                    transform=train_transform, valid=False, nfold=nfold),
                batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn, drop_last=False)

        valid_data_loader = DataLoader(
                dataset(csv_path, data_path,
                    transform=valid_transform, valid=True, nfold=nfold),
                batch_size=batch_size, shuffle=False, num_workers=8, worker_init_fn=worker_init_fn, drop_last=False)

    elif valid_mode == 'manual':
        train_data_loader = DataLoader(
                dataset(csv_path, data_path,
                    transform=train_transform, valid=False, nfold=0),
                batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn, drop_last=False)

        valid_data_loader = DataLoader(
                dataset(csv_path, data_path,
                    transform=valid_transform, valid=False, nfold=0),
                batch_size=batch_size, shuffle=False, num_workers=8, worker_init_fn=worker_init_fn, drop_last=False)
    else:
        print('valid mode {} is not implimented'.format(mode, valid))
    return train_data_loader, valid_data_loader
