import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from contrast_zstm.utils.data.data_handler import DataHandler
from contrast_zstm.networks.encoder import Encoder
from contrast_zstm.networks.decoder import Decoder


class ContrastZSTM:
    """
    Main class for contrastive zero-shot topic modeling.

    :param n_topics: int, number of topics to learn
    :param language1: str, the first language of the training corpus
    :param language2: str, the second language of the training corpus 
    (default "en")
    :param hidden_sizes: tuple, sizes for each hidden layer 
    (default (100, 100))
    :param n_epochs: int, number of epochs for training (default 20)
    :param learning_rate: float, learning rate for training 
    (default 2e-3)
    :param momentum: float, momentum for training (default 0.99)
    """

    def __init__(
            self,
            n_topics,
            language1, language2="english",
            embedding_model="distiluse-base-multilingual-cased-v1",
            embedding_dim=512,
            hidden_sizes=(100, 100),
            n_epochs=20,
            batch_size=64,
            learning_rate=2e-3,
            momentum=0.99,
            temperature=1):
        self.n_topics = n_topics
        self.language1 = language1
        self.language2 = language2
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.temperature = temperature

        if torch.cuda.is_available: self.device = torch.device("cuda")
        else: self.device = torch.device("cpu")

        # Model components
        self.data_handler = DataHandler(language1, language2, embedding_model)
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.similarity = nn.CosineSimilarity(dim=-1).to(self.device)
        self.train_losses = []
    
    def ingest_training(self, inputs1, inputs2):
        """
        Ingest a parallel corpus for training the ContrastZSTM model. 
        This will overwrite any prior corpus ingested by the model.

        :param inputs1: list, the corpus's documents in the first
        language
        :param inputs2: list, the corpus's documents in the second
        language
        """
        self.data_handler.clear_inputs()
        for pair in tuple(zip(inputs1, inputs2)):
            self.data_handler.add_parallel(pair)
        self.data_handler.clean()
        self.data_handler.embed()
        self.data_handler.bag()

    def init_vae(self):
        """
        Initialize the variational autoencoder. This should be done 
        after document ingest.
        """
        self.encoder = Encoder(
            self.embedding_dim, self.n_topics, self.hidden_sizes
        )
        self.decoder = Decoder(
            self.n_topics,
            self.data_handler.get_vocabulary_size(self.language1),
            self.data_handler.get_vocabulary_size(self.language2),
            self.language1,
            self.language2
        )

        self.optimizer = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
            betas=(self.momentum, 0.99)
        )

    def fit(self, n_workers=1):
        """
        Train the ContrastZSTM model on the ingested documents.

        :param n_workers: int, number of threads to use for loading data
        """
        train_data = self.data_handler.export_parallel()
        loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers,
            drop_last=True
        )

        samples_processed = 0
        progress_bar = tqdm(self.n_epochs, position=0, leave=True)
        for epoch in range(self.n_epochs):
            start_time = datetime.datetime.now()
            samples, train_loss = self._train_epoch(loader)
            samples_processed += samples
            self.train_losses.append(train_loss)
            end_time = datetime.datetime.now()
            progress_bar.update(1)
            progress_bar.set_description(
                """
                Epoch: [{}/{}]\tSeen Samples: [{}/{}]\tTrain Loss: {}\tTime: {}
                """.format(
                    epoch + 1, self.n_epochs,
                    samples_processed, len(train_data) * self.n_epochs,
                    train_loss,
                    end_time - start_time
                )
            )
        
        progress_bar.close()

    def get_topic_words(self, language, k=10):
        """
        Get the top k words in each topic for the passed language.

        :param language: string, the language to query
        :param k: int, number of topi words to retrieve for each topic
        """
        if language == self.language1:
            beta = self.decoder.beta1
            vocabulary = self.data_handler.vocabulary1
        elif language == self.language2:
            beta = self.decoder.beta2
            vocabulary = self.data_handler.vocabulary2
        
        topics = []
        for i in range(self.n_topics):
            _, indices = torch.topk(beta[i], k)
            words = [vocabulary[j] for j in indices.cpu().numpy()]
            topics.append(words)
        
        return topics

    @staticmethod
    def _rt(mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def _train_epoch(self, loader):
        # Put networks in training mode
        self.encoder.train()
        self.decoder.train()
        
        train_loss = 0
        samples_processed = 0

        for batch in loader:
            bow1 = batch["bow1"]
            bow2 = batch["bow2"]
            embedding1 = batch["embedding1"]
            embedding2 = batch["embedding2"]

            if torch.cuda.is_available():
                bow1 = bow1.cuda()
                bow2 = bow2.cuda()
                embedding1 = embedding1.cuda()
                embedding2 = embedding2.cuda()

            # Forward pass
            self.encoder.zero_grad()
            self.decoder.zero_grad()

            mu1, log_sigma1 = self.encoder(embedding1)
            mu2, log_sigma2 = self.encoder(embedding2)
            z1 = self._rt(mu1, log_sigma1)
            z2 = self._rt(mu2, log_sigma2)
            word_dist1 = self.decoder(z1, self.language1)
            word_dist2 = self.decoder(z2, self.language2)

            # Backward pass
            loss = self._loss(
                mu1, mu2,
                log_sigma1, log_sigma2,
                word_dist1, word_dist2,
                bow1, bow2,
                z1, z2
            ).sum()
            loss.backward()
            self.optimizer.step()

            # Compute training loss
            samples_processed += bow1.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss
    
    def _loss(
            self,
            mu1, mu2,
            log_sigma1, log_sigma2,
            word_dist1, word_dist2,
            bow1, bow2,
            z1, z2):
        prior_mu = self.decoder.prior_mu
        prior_sigma = self.decoder.prior_sigma
        sigma1 = torch.exp(log_sigma1)
        sigma2 = torch.exp(log_sigma2)
        
        KL1 = (torch.sum(sigma1 / prior_sigma, dim=1)
               + torch.sum(torch.square(prior_mu - mu1) / prior_sigma, dim=1)
               - self.n_topics
               + (prior_sigma.log().sum() - log_sigma1.sum(dim=1))) * 0.5
        KL2 = (torch.sum(sigma2 / prior_sigma, dim=1)
               + torch.sum(torch.square(prior_mu - mu2) / prior_sigma, dim=1)
               - self.n_topics
               + (prior_sigma.log().sum() - log_sigma2.sum(dim=1))) * 0.5
        
        RE1 = torch.sum(bow1 * torch.log(word_dist1 + 1e-10), dim=1)
        RE2 = torch.sum(bow2 * torch.log(word_dist2 + 1e-10), dim=1)

        CL = self._infoNCE(z1, z2, self.temperature)

        return (KL1 + KL2 - RE1 - RE2 - CL).sum()

    def _infoNCE(self, z1, z2, tau):
        sim11 = self.similarity(z1.unsqueeze(-2), z1.unsqueeze(-3)) / tau
        sim12 = self.similarity(z1.unsqueeze(-2), z2.unsqueeze(-3)) / tau
        sim22 = self.similarity(z2.unsqueeze(-2), z2.unsqueeze(-3)) / tau

        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float("-inf")
        sim22[..., range(d), range(d)] = float("-inf")
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)

        return nce_loss
