from __future__ import print_function

import codecs
from collections import Counter
import random

import numpy as np

from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import merge
from keras.models import Sequential, Model


def make_sampling_table(size, sampling_factor=1e-5):
    """
          This generates an array where the ith element
          is the probability that a word of rank i would be sampled,
          according to the sampling distribution used in word2vec.
          The word2vec formula is:
              p(word) = min(1, sqrt(word.frequency/sampling_factor) / (word.frequency/sampling_factor))
          We assume that the word frequencies follow Zipf's law (s=1) to derive
          a numerical approximation of frequency(rank):
             frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))
          where gamma is the Euler-Mascheroni constant.
            
         Parameters:
        -----------
        size: int, number of possible words to sample. 
    """

    gamma = 0.577
    rank = np.array(list(range(size)))
    rank[0] = 1
    inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1./(12.*rank)
    f = sampling_factor * inv_fq
    return np.minimum(1., f / np.sqrt(f))

def skipgrams(sequence, vocabulary_size, 
              window_size=4, negative_samples=1., shuffle=True, 
              categorical=False, sampling_table=None):
    """
         Paramaters:
         -----------
         vocabulary_size: int. maximum possible word index + 1
         window_size: int. actually half-window. The window of a word wi will be [i-window_size, i+window_size+1]
         negative_samples: float >= 0. 0 for no negative (=random) samples. 1 for same number as positive samples. etc.
         categorical: bool. if False, labels will be integers (eg. [0, 1, 1 .. ]),  if True labels will be categorical eg. [[1,0],[0,1],[0,1] .. ]

         Returns:
         --------
         couples, lables: where `couples` are int pairs and
             `labels` are either 0 or 1.
 
         Notes:
         ------
         By convention, index 0 in the vocabulary is a non-word and will be skipped.

    """
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i-window_size)
        window_end = min(len(sequence), i+window_size+1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0,1])
                else:
                    labels.append(1)

    return couples, labels


def build_model(vocab_size, embed_dim):
    pivot_inp = Input(shape=(1, ), dtype='int32', name='pivot')
    context_inp = Input(shape=(1, ), dtype='int32', name='context')

    pivot_embed = Embedding(input_dim=vocab_size, output_dim=embed_dim)(pivot_inp)
    context_embed = Embedding(input_dim=vocab_size, output_dim=embed_dim)(context_inp)

    prod = merge([pivot_embed, context_embed], mode='dot', dot_axes=2)
    res = Reshape((1, ), input_shape=(1, 1))(prod)

    activ = Activation('sigmoid', name='label')(res)

    model = Model(input=[pivot_inp, context_inp], output=activ)

    optim = RMSprop()
    model.compile(loss='mse', optimizer=optim)
    return model

def main():
    MAX_VOCAB = 10000
    WINDOW_SIZE = 4

    cutoff = 100000
    words = codecs.open('../data/Austen_Sense.txt', 'r', encoding='utf8').read().lower().split()[:cutoff]
    print('Loaded', len(words), 'words')
    cnt = Counter(words)
    print(cnt.most_common(30))
    indexer = {'UNK': 0}
    for w, c in cnt.most_common(MAX_VOCAB):
        indexer[w] = len(indexer)


    model = build_model(vocab_size=len(indexer), embed_dim=50)
    model.summary()

    sampling_table = make_sampling_table(size=len(indexer))

    idx = 0
    losses = []
    for idx in range(WINDOW_SIZE, len(words)-WINDOW_SIZE):
        seq = []
        for w in words[(idx - WINDOW_SIZE) : (idx + WINDOW_SIZE)]:
            try:
                seq.append(indexer[w])
            except KeyError:
                seq.append(0)
        
        couples, labels = skipgrams(seq, len(indexer), 
                                    window_size=4, negative_samples=1., shuffle=True, 
                                     categorical=False, sampling_table=sampling_table)

        if len(couples) > 1:
            couples = np.array(couples, dtype='int32')
            labels = np.array(labels, dtype='int32')

            p_inp = couples[:, 0]
            p_inp = p_inp[:, np.newaxis]

            c_inp = couples[:, 1]
            c_inp = c_inp[:, np.newaxis]
            
            loss = model.train_on_batch({'pivot': p_inp, 'context': c_inp}, {'label': labels})
            losses.append(loss)

            if idx % 1000 == 0:
                print(np.mean(losses))

if __name__ == '__main__':
    main()

