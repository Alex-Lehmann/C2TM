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


def reparameterize(mu, log_sigma):
    std = torch.exp(0.5 * log_sigma)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


class Decoder(nn.Module):
    def __init__(
            self,
            n_topics,
            vocabulary1_size, vocabulary2_size,
            language1, language2="english"):
        super(Decoder, self).__init__()
        self.n_topics = n_topics
        self.vocabulary1_size = vocabulary1_size
        self.vocabulary2_size = vocabulary2_size
        self.language1 = language1
        self.language2 = language2

        # Initialize priors
        self.prior_mu = torch.tensor([0.0] * n_topics)
        self.prior_sigma = torch.tensor([1.0 / (1.0 / n_topics)] * n_topics)
        if torch.cuda.is_available():
            self.prior_mu = self.prior_mu.cuda()
            self.prior_sigma = self.prior_sigma.cuda()
        self.prior_mu = nn.Parameter(self.prior_mu)
        self.prior_sigma = nn.Parameter(self.prior_sigma)

        # Initialize word-topic distributions
        self.beta1 = torch.Tensor(n_topics, vocabulary1_size)
        self.beta2 = torch.Tensor(n_topics, vocabulary2_size)
        if torch.cuda.is_available():
            self.beta1 = self.beta1.cuda()
            self.beta2 = self.beta2.cuda()
        self.beta1 = nn.Parameter(self.beta1)
        self.beta2 = nn.Parameter(self.beta2)
        nn.init.xavier_uniform_(self.beta1)
        nn.init.xavier_uniform_(self.beta2)

        self.beta1_batchnorm = nn.BatchNorm1d(vocabulary1_size, affine=False)
        self.beta2_batchnorm = nn.BatchNorm1d(vocabulary2_size, affine=False)

        self.dropout = nn.Dropout(p=0.2)

        self.topic_word_matrix1 = None
        self.topic_word_matrix2 = None
    
    def forward(self, z, language):
        theta = self.dropout(F.softmax(z, dim=1))
        if language == self.language1:
            beta = F.softmax(self.beta1_batchnorm(self.beta1), dim=1)
            self.topic_word_matrix1 = beta
            word_dist = torch.matmul(theta, beta)
        else:
            beta = F.softmax(self.beta2_batchnorm(self.beta2), dim=1)
            self.topic_word_matrix2 = beta
            word_dist = torch.matmul(theta, beta)
        
        return word_dist
