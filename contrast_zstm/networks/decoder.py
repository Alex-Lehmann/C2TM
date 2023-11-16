import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
            self,
            n_topics,
            vocabulary1_size,
            vocabulary2_size,
            language1,
            language2="english"):
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
            word_dist = F.softmax(
                self.beta1_batchnorm(torch.matmul(theta, self.beta1)), dim=1
            )
            self.topic_word_matrix1 = self.beta1
        else:
            word_dist = F.softmax(
                self.beta2_batchnorm(torch.matmul(theta, self.beta2)), dim=1
            )
            self.topic_word_matrix2 = self.beta2
        
        return word_dist
