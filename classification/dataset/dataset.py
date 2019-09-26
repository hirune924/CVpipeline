#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
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


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(worker_id)

def get_transforms(trans_list=[], trans_lib='torchvision', mode='yaml'):
    if trans_lib=='torchvision':
        lib = transforms
    elif trans_lib=='albumentations':
        lib = A

    if mode=='yaml':
        trans = [ getattr(lib, trans_dict['name'])(**trans_dict['params']) for trans_dict in trans_list]
    elif mode=='custom':
        if trans_lib=='torchvision':
            trans = [transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] 
        elif trans_lib=='albumentations':
            trans = [A.Resize(height=h, width=w, interpolation=1, always_apply=False, p=1),
                     A.Flip(always_apply=False, p=0.75),
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)]        
    return lib.Compose(trans)

class BasicDataset(Dataset):
    def __init__(self, csv_path, data_path, transform=None, trans_lib='torchvision', valid=False, nfold=0):
        self.data_path = data_path
        self.csv_file = csv_path
        self.data = pd.read_csv(self.csv_file)
        self.transform = transform
        self.trans_lib = trans_lib

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
        label = self.data.loc[idx, 'label']
        if self.trans_lib=='torchvision':
            image = Image.open(img_name)

            if self.transform is not None:
                image = self.transform(image)
        elif self.trans_lib=='albumentations':
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                image = self.transform(image=image)
            image = torch.from_numpy(image['image'].transpose(2, 0, 1))
            

        return image, label


class LMDBDataset(Dataset):
    def __init__(self, csv_path, lmdb_path, transform=None, trans_lib='torchvision', valid=False, nfold=0):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(self.lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        self.csv_file = csv_path
        self.data = pd.read_csv(self.csv_file)
        self.transform = transform
        self.trans_lib = trans_lib

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

        label = self.data.loc[idx, 'label']
        if self.trans_lib=='torchvision':
            image = Image.fromarray(cv2.cvtColor(cv2_row, cv2.COLOR_BGR2RGB))

            if self.transform is not None:
                image = self.transform(image)

        elif self.trans_lib=='albumentations':
            image = cv2.cvtColor(cv2_row, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                image = self.transform(image=image)
            image = torch.from_numpy(image['image'].transpose(2, 0, 1))

        return image, label


def load_data(csv_path, data_path, batch_size, train_trans_list=[], valid_trans_list=[], trans_mode='yaml', trans_lib='torchvision', valid_mode='auto', nfold=0, mode='basic', lmdb_path=None):
    if mode == 'basic':
        dataset = BasicDataset
    elif mode == 'lmdb':
        dataset = LMDBDataset
        data_path = lmdb_path

   
    train_transform = get_transforms(train_trans_list, trans_lib, trans_mode)
    valid_transform = get_transforms(valid_trans_list, trans_lib, trans_mode)
    if valid_mode == 'auto':
        train_data_loader = DataLoader(
                dataset(csv_path, data_path,
                    transform=train_transform, trans_lib=trans_lib, valid=False, nfold=nfold),
                batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn, drop_last=False)

        valid_data_loader = DataLoader(
                dataset(csv_path, data_path,
                    transform=valid_transform, trans_lib=trans_lib, valid=True, nfold=nfold),
                batch_size=batch_size, shuffle=False, num_workers=8, worker_init_fn=worker_init_fn, drop_last=False)

    elif valid_mode == 'manual':
        train_data_loader = DataLoader(
                dataset(csv_path, data_path,
                    transform=train_transform, trans_lib=trans_lib, valid=False, nfold=0),
                batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn, drop_last=False)

        valid_data_loader = DataLoader(
                dataset(csv_path, data_path,
                    transform=valid_transform, trans_lib=trans_lib, valid=False, nfold=0),
                batch_size=batch_size, shuffle=False, num_workers=8, worker_init_fn=worker_init_fn, drop_last=False)
    else:
        print('valid mode {} is not implimented'.format(mode, valid))
    return train_data_loader, valid_data_loader
