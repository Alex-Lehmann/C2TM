from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding_dim, n_topics, hidden_sizes=(100, 100)):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_topics = n_topics
        self.hidden_sizes = hidden_sizes

        # Network
        self.input = nn.Linear(embedding_dim, hidden_sizes[0])
        self.hiddens = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            in_dim = hidden_sizes[i]
            out_dim = hidden_sizes[i + 1]
            self.hiddens.append(nn.Linear(in_dim, out_dim))
        self.dropout = nn.Dropout(p=0.2)

        self.f_mu = nn.Linear(hidden_sizes[-1], n_topics)
        self.f_mu_batchnorm = nn.BatchNorm1d(n_topics, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], n_topics)
        self.f_sigma_batchnorm = nn.BatchNorm1d(n_topics, affine=False)

    def forward(self, x):
        x = F.softplus(self.input(x))
        for layer in self.hiddens:
            x = F.softplus(layer(x))
        x = self.dropout(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma

    def encode_single(self, x):
        x = F.softplus(self.input(x))
        for layer in self.hiddens:
            x = F.softplus(layer(x))
        x = self.dropout(x)
        mu = self.f_mu(x)
        log_sigma = self.f_sigma(x)

        return mu, log_sigma
