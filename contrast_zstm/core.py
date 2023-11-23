import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from contrast_zstm.data.data_handler import DataHandler
from contrast_zstm.networks.encoder import Encoder
from contrast_zstm.networks.decoder import Decoder
from contrast_zstm.training.regularizers import EarlyStopping


class ContrastZSTM:
    """
    Main class for contrastive zero-shot topic models.

    :param n_topics: int, number of topics to learn
    :param language1: str, the first language of the training corpus
    :param language2: str, the second language of the training corpus 
    (default "english")
    :param transformer_type: str, the sentence transformer to use for 
    document embedding
    :param embedding_dim: int, the dimension of the transformer's latent
    space
    :param hidden_sizes: tuple, sizes for each hidden layer 
    (default (100, 100))
    :param learning_rate: float, learning rate for training 
    (default 2e-3)
    :param momentum: float, momentum for training (default 0.99)
    :param temperature: float, temperature parameter for InfoNCE loss
    """

    def __init__(
            self,
            n_topics,
            language1, language2="english",
            transformer_type="distiluse-base-multilingual-cased-v1",
            embedding_dim=512,
            hidden_sizes=(100, 100),
            train_proportion=0.8,
            batch_size=64,
            learning_rate=2e-3,
            momentum=0.99,
            temperature=1.0):
        self.n_topics = n_topics
        self.language1 = language1
        self.language2 = language2
        self.transformer_type = transformer_type
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.train_proportion = train_proportion
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.temperature = temperature

        if torch.cuda.is_available(): self.device = torch.device("cuda")
        else: self.device = torch.device("cpu")

        # Model components
        self.data_handler = DataHandler(language1, language2, transformer_type)
        self.transformer = SentenceTransformer(transformer_type)
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.similarity = nn.CosineSimilarity(dim=-1).to(self.device)

        self.train_losses = []
        self.validation_losses = None

        self.validate = False
    
    def ingest_corpus(self, inputs1, inputs2):
        """
        Ingest a parallel corpus for training the ContrastZSTM model.

        :param inputs1: list, the corpus's documents in the first
        language
        :param inputs2: list, the corpus's documents in the second
        language
        """
        for pair in tuple(zip(inputs1, inputs2)):
            self.data_handler.add_pair(pair)
        self.data_handler.clean()
        self.data_handler.encode()
        self.data_handler.split(self.train_proportion)
        
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

    def fit(self, n_epochs=20, n_workers=1):
        """
        Train the ContrastZSTM model on the ingested documents.

        :param n_epochs: int, number of epochs for training (default 20)
        :param n_workers: int, number of threads to use for loading data
        """
        train_data = self.data_handler.export_training()
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers,
            drop_last=True
        )
        val_data = self.data_handler.export_validation()
        val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers,
            drop_last=True
        )
        stopper = EarlyStopping()

        samples_processed = 0
        progress_bar = tqdm(
            desc="Training progress", total=n_epochs, initial=1
        )
        for epoch in range(n_epochs):
            train_samples, train_loss = self._train_epoch(train_loader)
            samples_processed += train_samples
            self.train_losses.append(train_loss)
            progress_bar.update(1)

            _, val_loss = self._validation_epoch(val_loader)

            progress_bar.set_description(
                "Epoch: [{}/{}]\t\tTrain loss: {}\tValidation loss: {}".format(
                    epoch + 1,
                    n_epochs,
                    round(train_loss, 4),
                    round(val_loss, 4)
                )
            )

            if stopper(val_loss, self):
                progress_bar.close()
                print("Early stopping after {} epochs".format(epoch + 1))
                return stopper.checkpoint_model
        progress_bar.close()
        return self

    def get_topic_words(self, language, k=10):
        """
        Get the top k words for each topic in the passed language.

        :param language: string, the language to query
        :param k: int, number of top words to retrieve for each topic
        """
        if language == self.language1:
            phi = self.decoder.topic_word_matrix1
            vocabulary = self.data_handler.vocabulary1
        elif language == self.language2:
            phi = self.decoder.topic_word_matrix2
            vocabulary = self.data_handler.vocabulary2
        
        topics = []
        for i in range(self.n_topics):
            _, indices = torch.topk(phi[i], k)
            words = [vocabulary[j] for j in indices]
            weights = [phi[i, j].item() for j in indices]

            dict = {words[j]: weights[j] for j in range(len(words))}
            topics.append(dict)
        
        return tuple(topics)

    def predict_topics(self, document):
        """
        Predict the topic distribution in the passed document.

        :param document: string, a document
        """
        embedding = self.transformer.encode(document)
        mu, _ = self.encoder.encode_single(torch.Tensor(embedding))
        theta = F.softmax(mu, dim=0)

        return theta
    
    def predict_keywords(self, document, language, k=5):
        """
        Predict the top k most likely words for the topic mixture in the
        passed document. Words are in the passed language.

        :param document: string, a document
        :param language: string, the language in which to return words
        :param k: int, number of top words to retrieve
        """
        theta = self.predict_topics(document)
        if language == self.language1:
            phi = self.decoder.topic_word_matrix1
            vocabulary = self.data_handler.vocabulary1
        elif language == self.language2:
            phi = self.decoder.topic_word_matrix2
            vocabulary = self.data_handler.vocabulary2

        document_word_dist = torch.matmul(theta, phi)
        _, indices = torch.topk(document_word_dist, k)
        keywords = [vocabulary[i] for i in indices]

        return keywords

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
    
    def _validation_epoch(self, loader):
        # Put networks in inference mode
        self.encoder.eval()
        self.decoder.eval()
        
        val_loss = 0
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

            # Compute training loss
            samples_processed += bow1.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss
    
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

        return (KL1 + KL2 - RE1 - RE2 + CL).sum()

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
