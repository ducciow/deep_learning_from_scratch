from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv
import numpy as np
import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                 tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(mean=train_mean, std=train_std)])
        self._augment_transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                         tv.transforms.AutoAugment(),
                                                         tv.transforms.ToTensor(),
                                                         tv.transforms.Normalize(mean=train_mean, std=train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # read dataframe
        img_name = self.data.iloc[index, 0]
        img_label = self.data.iloc[index, 1:]

        # read image
        img_path = Path.cwd()/img_name
        img = imread(img_path)

        # transform image
        img = gray2rgb(img)
        if self.mode == 'train':
            img = self._augment_transform(img)
        else:
            img = self._transform(img)

        return img, torch.Tensor(img_label)
