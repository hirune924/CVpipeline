#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import random
from PIL import Image


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(worker_id)

class TrainDataset(Dataset):
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


def load_train_data(csv_path, data_path, batch_size, valid_mode='auto', nfold=0):
    dataset = TrainDataset
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
        print('valid mode {} is not implimented'.format(valid))
    return train_data_loader, valid_data_loader
