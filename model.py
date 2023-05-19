import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan

class encoder(nn.Module):
    def __init__(self,
                 label_embeddings,
                 input_dims,
                 h1_dims=300,
                 h2_dims=300,
                 h3_dims=300,
                 h4_dims=768,
                 batch_size=16,
                 lr=0.001,
                 max_train_length=None,
                 device='cuda'):
        super().__init__()
        self.input_dims = input_dims
        self.h1_dims = h1_dims
        self.h2_dims = h2_dims
        self.h3_dims = h3_dims
        self.h4_dims = h4_dims
        self.label_embeddings = label_embeddings
        self.input_h1_fc = self.fc_layer(input_dims, h1_dims)
        self.h1_h2_fc = self.fc_layer(h1_dims, h2_dims)
        self.h2_h3_fc = self.fc_layer(h2_dims, h3_dims)
        self.h4_fc = self.fc_layer(h1_dims + h2_dims + h3_dims, h4_dims)
        self.similarity = torch.dot
        self.softmax = torch.nn.Softmax()
        self.batch_size = batch_size
        self.lr = lr
        self.max_train_length = max_train_length,
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def fc_layer(self, in_dims, out_dims):
        return nn.Sequential(
            nn.Linear(in_dims, out_dims),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_dims)
        )

    def forward(self, x, y):
        l1 = self.input_h1_fc(x)
        l2 = self.h1_h2_fc(l1)
        l3 = self.h2_h3_fc(l2)
        l4 = self.h4_fc(torch.cat((l1, l2, l3), dim=1))
        sc = self.similarity(l4, self.embeddings)
        logits = self.softmax(sc)
        return logits, l4

    def predict(self):
        pass
        # TODO

    def loss(self, logits, l4, y, e):
        mse = self.mse_loss(l4, e)
        ce = self.ce_loss(logits, nn.functional.one_hot(y))
        return mse + ce

    def fit(self, train_data, train_labels, train_embeddings, n_epochs=None, n_iters=None, verbose=False):
        '''
        Training the model.
        returns loss: a list containing the training losses on each epoch.
        '''

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float),
                                      torch.from_numpy(train_labels).to(torch.float),
                                      torch.from_numpy(train_embeddings).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True,
                                  drop_last=True)

        optimizer = optim.Adam(self.parameters(), weight_decay=0.05)

        loss_log = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            for batch_x, batch_y, batch_e in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                optimizer.zero_grad()
                logits, l4 = self.forward(batch_x, batch_y)

                loss = self.loss(logits, l4, batch_y, batch_e)

                loss.backward()
                optimizer.step()

                cum_loss += loss.item()
                n_epoch_iters += 1

                self.n_iters += 1

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        return loss_log
