import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np


class encoder(nn.Module):
    def __init__(self, input_dims, h1_dims, h2_dims, h3_dims, h4_dims, embeddings):
        super().__init__()
        self.input_dims = input_dims
        self.h1_dims = h1_dims
        self.h2_dims = h2_dims
        self.h3_dims = h3_dims
        self.h4_dims = h4_dims
        self.embeddings = embeddings
        self.input_h1_fc = self.fc_layer(input_dims, h1_dims)
        self.h1_h2_fc = self.fc_layer(h1_dims, h2_dims)
        self.h2_h3_fc = self.fc_layer(h2_dims, h3_dims)
        self.h4_fc = self.fc_layer(h1_dims + h2_dims + h3_dims, h4_dims)
        self.similarity = torch.dot()
        self.optimizer = optim.Adam(self.parameters(), weight_decay = 0.05)


    def fc_layer(self, in_dims, out_dims):
        return nn.Sequential(
            nn.Linear(in_dims, out_dims),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_dims)
        )

    def forward(self, x, y, e):
        l1 = self.input_h1_fc(x)
        l2 = self.h1_h2_fc(l1)
        l3 = self.h2_h3_fc(l2)
        l4 = self.h4_fc(torch.cat((l1, l2, l3), dim=1))
        # Todo dot product should be with all classes, might have to loop or matrix multiply, after which we can take argmax
        sc = self.similarity(l4, e)
        # Todo: Will have to set axis for the softmax
        sc = nn.softmax(sc)
        # Todo: Check if the dimensions for this sc is correct
        return sc, l4

    def loss(self, l4, y, e):
        loss1 = nn.MSELoss()
        # We'll have to use one hot encodings for calculating Cross Entropy loss
        loss2 = nn.CrossEntropyLoss()
        return loss1(l4, e) #+ add CE loss 
    
    def predict(self):
        pass
        #TODO
