import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import string

class Word2Vec:
    def __init__(self, embedd_dim=100):
        self.vocab = None
        self.vocab_size = None
        self.word2idx = None
        self.ix2word = None
        self.alpha = 0.01
        self.embedd_dim = embedd_dim

    def create_vocabulary(self, corpus):
        self.vocab = sorted(set([word for sentence in corpus for word in sentence]))
        self.vocab_size = len(self.vocab)
        self.word2idx = dict((w,i) for i, w in enumerate(self.vocab))
        self.idx2word = np.array(self.vocab)

    def clean_corpus(self, corpus):
        """
         This function takes in a list of sentences, clean and tokenize them.
        """
        cleaned_corpus = []
        for sentence in corpus:
            cleaned_sentence = []
            for word in sentence.split():
                cleaned_word = self.clean_word(word)
                if not any(ch.isdigit() for ch in word):
                    cleaned_sentence.append(cleaned_word)
            cleaned_corpus.append(cleaned_sentence)
        self.create_vocabulary(cleaned_corpus)
        print('Corpus cleaned.')
        return cleaned_corpus

    def create_dataset(self, corpus, window_size=4):
        dataset = []
        for sentence in corpus:
            indicies = [self.word2idx[word] for word in sentence]
            for target in range(len(indicies)):
                for w in range(-window_size, window_size+1):
                    context = target + w
                    if context < 0 or context >= len(indicies) or context == target:
                        continue
                    context_idx = indicies[context]
                    target_idx = indicies[target]
                    dataset.append([target_idx, context_idx])

        return np.array(dataset)

    def clean_word(self, word):
        remove_punc = word.translate(str.maketrans('', '', string.punctuation))
        return remove_punc.lower()

    def generator(self, dataset, batch_size):
        X, Y = dataset[:,0], dataset[:, 1]
        n_samples = X.shape[0]

        indicies = np.arange(n_samples)
        np.random.shuffle(indicies)

        for start in range(0, n_samples, batch_size):
            end = min(start+batch_size, n_samples)
            batch_idx = indicies[start:end]
            yield np.concatenate((X[batch_idx].reshape(-1,1), Y[batch_idx].reshape(-1,1)), axis=1)

    def initialize_weights(self):
        self.W1 = Variable(torch.rand(self.vocab_size, self.embedd_dim), requires_grad=True)
        self.W2 = Variable(torch.rand(self.embedd_dim, self.vocab_size), requires_grad=True)

    def create_one_hot(self, word_ind):
        x = torch.zeros(self.vocab_size).float()
        x[word_ind] = 1.0
        return x

    def train(self, corpus, epochs=1):
        cleaned_corpus = self.clean_corpus(corpus)
        dataset = self.generator(self.create_dataset(cleaned_corpus), 32)
        self.initialize_weights()
        for ep in range(epochs):
            loss_val = 0
            for data, target in next(dataset):
                x = Variable(self.create_one_hot(data)).float()
                y = Variable(torch.from_numpy(np.array([target])).long())

                z1 = torch.matmul(self.W1.T, x)
                z2 = torch.matmul(self.W2.T, z1)

                softmax = F.log_softmax(z2, dim=0)
                loss = F.nll_loss(softmax.view(1,-1), y)
                loss_val += loss.data

                loss.backward()

                self.W1.data -= self.alpha * self.W1.grad.data
                self.W2.data -= self.alpha * self.W2.grad.data

                self.W1.grad.zero_()
                self.W2.grad.zero_()

            if (ep%100) == 0:
                print(f'ep: {ep} loss: {loss_val}')
