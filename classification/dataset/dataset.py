#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
from PIL import Image


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(worker_id)

class TrainDataset(Dataset):
    def __init__(self, csv_path, data_path, transform=None):
        self.data_path = data_path
        self.csv_file = csv_path
        self.data = pd.read_csv(self.csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, self.data.loc[idx, 'image'])
        image = Image.open(img_name)

        label = self.data.loc[idx, 'label']

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_train_data(csv_path, data_path, batch_size):
    dataset = TrainDataset
    data_loader = DataLoader(
            dataset(csv_path, data_path,
                transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
            batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn, drop_last=False)

    return data_loader
