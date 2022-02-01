import numpy as np
import torch
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # -reset the gradients.
        self._optim.zero_grad()
        # -propagate through the network
        y_pred = self._model(x)
        # -calculate the loss
        loss = self._crit(y_pred, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss

    def val_test_step(self, x, y):
        # propagate through the network and calculate the loss and predictions
        y_pred = self._model(x)
        loss = self._crit(y_pred, y)
        y_pred = torch.where(y_pred > 0.5, 1.0, 0.0)  # convert to one-hot
        # return the loss and the predictions
        return loss, y_pred

    def train_epoch(self):
        # set training mode
        self._model.train()
        # iterate through the training set
        total_loss = []
        for x, y in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            # perform a training step
            loss = self.train_step(x, y)
            total_loss.append(loss)
        # calculate the average loss for the epoch and return it
        avg_loss = torch.mean(torch.tensor(total_loss, dtype=torch.float))
        return avg_loss

    def val_test(self):
        # set eval mode
        # disable gradient computation
        self._model.eval()

        # iterate through the validation set
        y_true = []
        y_preds = []
        total_loss = []
        for x, y in self._val_test_dl:
            # transfer the batch to the gpu if given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            # perform a validation step
            loss, y_pred = self.val_test_step(x, y)

            # save the predictions and the labels for each batch
            total_loss.append(loss)

            y_preds.extend(y_pred.tolist())
            y_true.extend(y.tolist())

        # calculate the average loss and average metrics
        avg_loss = torch.mean(torch.tensor(total_loss, dtype=torch.float))
        f1 = f1_score(y_true, y_preds, average='micro')

        # return the loss and print the calculated metrics
        return avg_loss, f1

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_loss = []
        val_loss = []
        f1_scores = []
        num_epoch = 0
        best_score = 0
        plateau = 0  # number of epochs the val loss did not decrease
        # while True:
        for e in tqdm(range(epochs)):
            # stop by epoch number
            if num_epoch > epochs:
                break
            num_epoch += 1
            # train for a epoch and then calculate the loss and metrics on the validation set
            loss_ep = self.train_epoch()
            loss_v, f1 = self.val_test()

            # append the losses to the respective lists
            train_loss.append(loss_ep)
            val_loss.append(loss_v)
            f1_scores.append(f1)

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if f1 > best_score:
                best_score = f1
                self.save_checkpoint(num_epoch)

            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if self._early_stopping_patience > 0 and num_epoch > 1:
                if val_loss[-1] < val_loss[-2]:
                    plateau = 0
                else:
                    plateau += 1
                if plateau >= self._early_stopping_patience:
                    break
        # return the losses for both training and validation
        return train_loss, val_loss
