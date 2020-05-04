import numpy as np
from word2vec import Word2Vec

text = open('DATASET/simpsons.txt').read()
corpus = text.split('\n')

word2vec = Word2Vec()
word2vec.train(corpus)
