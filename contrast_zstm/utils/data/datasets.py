import torch
from torch.utils.data import Dataset

import scipy.sparse


class ParallelCorpus(Dataset):
    """
    Class to load bags-of-words and contextualized embeddings for 
    parallel sentence corpora. Used for training the ContrastZSTM.
    """

    def __init__(
            self,
            embeddings1, embeddings2,
            bow1, bow2,
            tokens1, tokens2):
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.bow1 = bow1
        self.bow2 = bow2
        self.tokens1 = tokens1
        self.tokens2 = tokens2
    
    def __len__(self):
        return self.bow1.shape[0]
    
    def __getitem__(self, i):
        embedding1 = torch.FloatTensor(self.embeddings1[i])
        embedding2 = torch.FloatTensor(self.embeddings2[i])

        bow1 = self.bow1[i]
        bow2 = self.bow2[i]
        if type(bow1) == scipy.sparse.csr_matrix: bow1 = bow1.todense()
        if type(bow2) == scipy.sparse.csr_matrix: bow2 = bow2.todense()
        bow1 = torch.FloatTensor(bow1).squeeze()
        bow2 = torch.FloatTensor(bow2).squeeze()

        return {"embedding1": embedding1, "embedding2": embedding2,
                "bow1": bow1, "bow2": bow2}


class MonoCorpus(Dataset):
    """
    Class to load bags-of-words and contextualized embeddings for
    monolingual sentence corpora.
    """

    def __init__(self, embeddings, bow, tokens):
        self.embeddings = embeddings
        self.bow = bow
        self.tokens = tokens

    def __len__(self):
        return self.bow.shape[0]
    
    def __getitem__(self, i):
        embedding = torch.FloatTensor(self.embeddings[i])
        
        bow = self.bow
        if type(bow) == scipy.sparse.csr_matrix: bow = bow.todense()
        bow = torch.FloatTensor(bow)

        return {"embedding": embedding, "bow": bow}
