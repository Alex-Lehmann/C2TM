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
            vocabulary1, vocabulary2):
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.bow1 = bow1
        self.bow2 = bow2
        self.vocabulary1 = vocabulary1
        self.vocabulary2 = vocabulary2
    
    def __len__(self):
        return len(self.bow1)
    
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
