from __future__ import print_function

import codecs
from collections import Counter
import random

import numpy as np

from keras.layers import Input
from keras.layers.core import Activation, Reshape, Flatten, Dense
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import merge
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers.recurrent import LSTM


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
         Parameters:
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


def build_model(vocab_size, embed_dim=50, level='word', token_len=15, token_char_vector_dict={},
                nb_recurrent_layers=3):
    if level == 'word':
        pivot_inp = Input(shape=(1, ), dtype='int32', name='pivot')
        pivot_embed = Embedding(input_dim=vocab_size, output_dim=embed_dim)(pivot_inp)

    elif level == 'char':
        pivot_inp = Input(shape=(token_len, len(token_char_vector_dict)),
                                 name='pivot')
        for i in range(nb_recurrent_layers):
            if i == 0:
                curr_input = pivot_inp
            else:
                curr_input = curr_out

            l2r = LSTM(output_dim=embed_dim,
                           return_sequences=True,
                           activation='tanh')(curr_input)
            r2l = LSTM(output_dim=embed_dim,
                           return_sequences=True,
                           activation='tanh',
                           go_backwards=True)(curr_input)
            curr_out = merge([l2r, r2l], name='encoder_'+str(i+1), mode='sum')

        flattened = Flatten()(curr_out)
        pivot_embed = Dense(embed_dim)(flattened)
        pivot_embed = Reshape((1, embed_dim))(pivot_embed)

    context_inp = Input(shape=(1, ), dtype='int32', name='context')
    context_embed = Embedding(input_dim=vocab_size, output_dim=embed_dim)(context_inp)
    
    prod = merge([pivot_embed, context_embed], mode='dot', dot_axes=2)
    res = Reshape((1, ), input_shape=(1, 1))(prod)

    activ = Activation('sigmoid', name='label')(res)

    model = Model(input=[pivot_inp, context_inp], output=activ)

    optim = RMSprop()
    model.compile(loss='mse', optimizer=optim)
    return model

def index_characters(tokens):
    vocab = {ch for tok in tokens for ch in tok.lower()}

    vocab = vocab.union({'$', '|', '%'})

    char_vocab = tuple(sorted(vocab))
    char_vector_dict, char_idx = {}, {}
    filler = np.zeros(len(char_vocab), dtype='float32')

    for idx, char in enumerate(char_vocab):
        ph = filler.copy()
        ph[idx] = 1
        char_vector_dict[char] = ph
        char_idx[idx] = char

    return char_vector_dict, char_idx

def vectorize_token(seq, char_vector_dict, max_len):
    # cut, if needed:
    seq = seq[:(max_len - 2)]
    seq = '%' + seq + '|'
    seq = seq[::-1] # reverse order (cf. paper)!

    filler = np.zeros(len(char_vector_dict), dtype='float32')

    seq_X = []
    for char in seq:
        try:
            seq_X.append(char_vector_dict[char])
        except KeyError:
            seq_X.append(filler)
    
    while len(seq_X) < max_len:
        seq_X.append(filler)
    
    return np.array(seq_X, dtype='float32')

def vectorize_tokens(tokens, char_vector_dict, max_len=15):
    X = []
    for token in tokens:
        token = token.lower()
        x = vectorize_token(seq=token,
                            char_vector_dict=char_vector_dict,
                            max_len=max_len)
        X.append(x)

    return np.asarray(X, dtype='float32')

def main():
    MAX_VOCAB = 10000
    WINDOW_SIZE = 4
    LEVEL = 'char'
    MAX_TOKEN_LEN = 15
    NB_LAYERS = 1

    cutoff = 100000
    words = codecs.open('../data/Austen_Sense.txt', 'r', encoding='utf8').read().lower().split()[:cutoff]
    print('Loaded', len(words), 'words')

    cnt = Counter(words)
    print(cnt.most_common(30))

    word_to_int = {'UNK': 0}
    for w, c in cnt.most_common(MAX_VOCAB):
        word_to_int[w] = len(word_to_int)
    int_to_word = [None] * len(word_to_int)
    for k, v in word_to_int.items():
        int_to_word[v] = k

    if LEVEL == 'char':
        char_vector_dict, char_idx = index_characters(int_to_word)
        print(char_vector_dict.keys())

    model = build_model(vocab_size=len(word_to_int),
                        embed_dim=50,
                        level=LEVEL,
                        token_len=MAX_TOKEN_LEN,
                        token_char_vector_dict=char_vector_dict,
                        nb_recurrent_layers=NB_LAYERS)
    model.summary()

    sampling_table = make_sampling_table(size=len(word_to_int))

    idx = 0
    losses = []

    for idx in range(WINDOW_SIZE, len(words)-WINDOW_SIZE):
        seq = []
        for w in words[(idx - WINDOW_SIZE) : (idx + WINDOW_SIZE)]:
            try:
                seq.append(word_to_int[w])
            except KeyError:
                seq.append(0)
        
        couples, labels = skipgrams(seq, len(word_to_int), 
                                    window_size=4, negative_samples=1., shuffle=True, 
                                     categorical=False, sampling_table=sampling_table)

        if len(couples) > 1:
            print('.', end='')
            couples = np.array(couples, dtype='int32')

            c_inp = couples[:, 1]
            c_inp = c_inp[:, np.newaxis]

            if LEVEL == 'word':
                p_inp = couples[:, 0]
                p_inp = p_inp[:, np.newaxis]
            elif LEVEL == 'char':
                tokens = [int_to_word[i] for i in couples[:, 0]]
                p_inp = vectorize_tokens(tokens=tokens,
                                         char_vector_dict=char_vector_dict,
                                         max_len=MAX_TOKEN_LEN)
            else:
                raise ValueError('Wrong level param: word or char')

            labels = np.array(labels, dtype='int32')
            
            loss = model.train_on_batch({'pivot': p_inp, 'context': c_inp}, {'label': labels})
            losses.append(loss)

            if idx % 1000 == 0:
                print(np.mean(losses))

"""
# recover the embedding weights trained with skipgram:
weights = model.layers[0].get_weights()[0]

# we no longer need this
del model

weights[:skip_top] = np.zeros((skip_top, dim_proj))
norm_weights = np_utils.normalize(weights)

word_index = tokenizer.word_index
reverse_word_index = dict([(v, k) for k, v in list(word_index.items())])
word_index = tokenizer.word_index

def embed_word(w):
    i = word_index.get(w)
    if (not i) or (i<skip_top) or (i>=max_features):
        return None
    return norm_weights[i]

def closest_to_point(point, nb_closest=10):
    proximities = np.dot(norm_weights, point)
    tups = list(zip(list(range(len(proximities))), proximities))
    tups.sort(key=lambda x: x[1], reverse=True)
    return [(reverse_word_index.get(t[0]), t[1]) for t in tups[:nb_closest]]  

def closest_to_word(w, nb_closest=10):
    i = word_index.get(w)
    if (not i) or (i<skip_top) or (i>=max_features):
        return []
    return closest_to_point(norm_weights[i].T, nb_closest)


''' the resuls in comments below were for: 
    5.8M HN comments
    dim_proj = 256
    nb_epoch = 2
    optimizer = rmsprop
    loss = mse
    max_features = 50000
    skip_top = 100
    negative_samples = 1.
    window_size = 4
    and frequency subsampling of factor 10e-5. 
'''

words = ["article", # post, story, hn, read, comments
"3", # 6, 4, 5, 2
"two", # three, few, several, each
"great", # love, nice, working, looking
"data", # information, memory, database
"money", # company, pay, customers, spend
"years", # ago, year, months, hours, week, days
"android", # ios, release, os, mobile, beta
"javascript", # js, css, compiler, library, jquery, ruby
"look", # looks, looking
"business", # industry, professional, customers
"company", # companies, startup, founders, startups
"after", # before, once, until
"own", # personal, our, having
"us", # united, country, american, tech, diversity, usa, china, sv
"using", # javascript, js, tools (lol)
"here", # hn, post, comments
]

for w in words:
    res = closest_to_word(w)
    print('====', w)
    for r in res:
        print(r)
"""

if __name__ == '__main__':
    main()

