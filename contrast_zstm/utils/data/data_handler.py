import string

import nltk
from nltk.corpus import stopwords
from gensim.utils import deaccent
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

from contrast_zstm.utils.data.datasets import ParallelCorpus

class DataHandler:
    def __init__(
            self,
            language1, language2,
            embedding_model="distiluse-base-multilingual-cased-v1",
            vocabulary_size=2000):
        self.language1 = language1
        self.language2 = language2
        self.embedding_model = embedding_model
        self.vocabulary_size = vocabulary_size

        self.input_docs1 = []
        self.input_docs2 = []
        self.raw_docs1 = None
        self.raw_docs2 = None
        self.processed_docs1 = None
        self.processed_docs2 = None
        self.embeddings1 = None
        self.embeddings2 = None
        self.bow1 = None
        self.bow2 = None
        self.vocabulary1 = None
        self.vocabulary2 = None

        nltk.download("stopwords")
        self.stop_words = (list(stopwords.words(language1)),
                           list(stopwords.words(language2)))
    
    def add_parallel(self, pair):
        self.input_docs1.append(pair[0])
        self.input_docs2.append(pair[1])
    
    def clear_inputs(self):
        self.input_docs1 , self.input_docs2 = [], []
    
    def clean(self, vocabulary_size=2000, min_words=1):
        tmp_docs, vocabularies, retained_indices = [], [], []
        for i in (0, 1):
            docs = (self.input_docs1, self.input_docs2)[i]

            docs = [deaccent(doc.lower()) for doc in docs]
            docs = [doc.translate(str.maketrans(
                string.punctuation, " " * len(string.punctuation)
            )) for doc in docs]
            docs = [doc.translate(str.maketrans(
                "0123456789", " " * len("0123456789")
            )) for doc in docs]
            docs = [" ".join([w for w in doc.split() if len(w) > 0
                              and w not in self.stop_words[i]])
                              for doc in docs]
            
            vectorizer = CountVectorizer(max_features=vocabulary_size)
            vectorizer.fit_transform(docs)
            vocabulary = set(vectorizer.get_feature_names_out())
            docs = [" ".join([w for w in doc.split() if w in vocabulary])
                    for doc in docs]
            
            tmp_docs.append(docs)
            vocabularies.append(vocabulary)
            retained_indices.append([i for i in range(len(docs))
                                     if len(docs[i]) >= min_words])
        
        shared_indices = [i for i in retained_indices[0]
                          if i in retained_indices[1]]
        self.raw_docs1 = tuple([self.input_docs1[i] for i in shared_indices])
        self.raw_docs2 = tuple([self.input_docs2[i] for i in shared_indices])
        self.processed_docs1 = tuple([tmp_docs[0][i] for i in shared_indices])
        self.processed_docs2 = tuple([tmp_docs[1][i] for i in shared_indices])
    
    def embed(self, batch_size=200):
        model = SentenceTransformer(self.embedding_model)
        self.embeddings1 = model.encode(self.raw_docs1, batch_size=batch_size)
        self.embeddings2 = model.encode(self.raw_docs2, batch_size=batch_size)

    def bag(self):
        vectorizer1 = CountVectorizer()
        vectorizer2 = CountVectorizer()

        self.bow1 = vectorizer1.fit_transform(self.processed_docs1)
        self.bow2 = vectorizer2.fit_transform(self.processed_docs2)
        self.vocabulary1 = vectorizer1.get_feature_names_out()
        self.vocabulary2 = vectorizer2.get_feature_names_out()
    
    def export_parallel(self):
        return ParallelCorpus(self.embeddings1, self.embeddings2,
                              self.bow1, self.bow2,
                              self.vocabulary1, self.vocabulary2)

    def get_embedding_dim(self):
        return self.embeddings1[0].shape[0]
    
    def get_vocabulary_size(self, language):
        if language == self.language1: return len(self.vocabulary1)
        elif language == self.language2: return len(self.vocabulary2)
