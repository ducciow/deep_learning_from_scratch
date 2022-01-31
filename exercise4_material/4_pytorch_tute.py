import numpy as np
import torch
import torchvision as tv
from torch import nn
from torch.nn import functional as F
import operator
import time

data = tv.datasets.FashionMNIST(root='data', download=True)
batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(data))), 100, False)
gpu = torch.cuda.is_available()


class Timer:
    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, *args):
        self.end = time.process_time()
        self.interval = self.end - self.start


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 100, 5)
        self.conv2 = nn.Conv2d(100, 50, 5)
        self.conv3 = nn.Conv2d(50, 5, 5)

        self.fc1 = nn.Linear(1280, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.fc1(x.flatten(start_dim=1)))
        return self.fc2(x)


model = Model()
if gpu:
    model.cuda()
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())

with Timer() as t:
    for e in range(1):
        for idx in batch_sampler:
            batch = operator.itemgetter(*idx)(data)

            x = []
            y_true = []
            for img, label in batch:
                x.append(np.asarray(img))
                y_true.append(label)

            x = np.array(x)[:, np.newaxis]
            y_true = np.array(y_true)
            if gpu:
                x = torch.tensor(x, dtype=torch.float).cuda()
                y_true = torch.tensor(y_true, dtype=torch.long).cuda()
            else:
                x = torch.tensor(x, dtype=torch.float)
                y_true = torch.tensor(y_true, dtype=torch.long)

            y_pred = model(x)

            l = loss(y_pred, y_true)

            print(l)

            optim.zero_grad()
            l.backward()
            optim.step()

print(t.interval)
