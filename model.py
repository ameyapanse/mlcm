import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class encoder(nn.Module):
    def __init__(self, input_dims, h1_dims, h2_dims, h3_dims, h4_dims):
        super().__init__()
        self.input_dims = input_dims
        self.h1_dims = h1_dims
        self.h2_dims = h2_dims
        self.h3_dims = h3_dims
        self.h4_dims = h4_dims
        self.input_h1_fc = self.fc_layer(input_dims, h1_dims)
        self.h1_h2_fc = self.fc_layer(h1_dims, h2_dims)
        self.h2_h3_fc = self.fc_layer(h2_dims, h3_dims)
        self.h4_fc = self.fc_layer(h1_dims + h2_dims + h3_dims, h4_dims)
        self.similarity = nn.CosineSimilarity()

    def fc_layer(self, in_dims, out_dims):
        return nn.Sequential(
            nn.Linear(in_dims, out_dims),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_dims)
        )

    def forward(self, x, y, activity_word_embedding):
        l1 = self.input_h1_fc(x)
        l2 = self.h1_h2_fc(l1)
        l3 = self.h2_h3_fc(l2)
        l4 = self.h4_fc(torch.cat((l1, l2, l3), dim=1))
        sim = self.similarity(l4, activity_word_embedding)


