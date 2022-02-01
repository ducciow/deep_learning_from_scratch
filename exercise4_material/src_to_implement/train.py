import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCH = 10
DATA_4_DEBUG = 500

# load the data from the csv file and perform a train-test-split
csv_path = Path('./data.csv')
csv_df = pd.read_csv(csv_path, sep=';')

csv_df = csv_df.iloc[:DATA_4_DEBUG]

training_set, val_st = train_test_split(csv_df, test_size=0.3)

# set up data loading for the training and validation set
training_data = t.utils.data.DataLoader(ChallengeDataset(training_set, 'train'), batch_size=BATCH_SIZE)
val_data = t.utils.data.DataLoader(ChallengeDataset(val_st, 'val'), batch_size=BATCH_SIZE)

# create an instance of our ResNet model
m = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
loss_fn = t.nn.BCELoss()

# set up the optimizer
optim = t.optim.Adam(m.parameters(), lr=LEARNING_RATE)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(m,  # Model to be trained.
                  loss_fn,  # Loss function
                  optim=optim,  # Optimizer
                  train_dl=training_data,  # Training data set
                  val_test_dl=val_data,  # Validation (or test) data set
                  cuda=True,  # Whether to use the GPU
                  early_stopping_patience=5)

# go, go, go... call fit on trainer
res = trainer.fit(NUM_EPOCH)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
