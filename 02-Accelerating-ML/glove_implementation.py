# -*- coding: utf-8 -*-
"""GloVe implementation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eCUkT-fn6oUiRz3utDCy9zPsVT6uc51e

# Custom Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import itertools
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

nltk.download('brown')
nltk.download('stopwords')
stopwords = stopwords.words('english')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bool_train = True

# info

# number of training epochs
n_epochs = 4000

# tolerance
eps = 0.001

# number of sentences to consider
n_sents = 100

# weight embedding size
embedding_size = 20

# learning rate
alpha = 0.1

# AdaGrad parameter
delta = 0.8

# context_window_size
window_size = 5

# top N similar words
topN = 5

brown = nltk.corpus.brown
sents = brown.sents()[:n_sents]

print('Processing sentences..\n')
processed_sents = []
for sent in sents:
    processed_sents.append([word.lower() for word in sent if word.isalnum() and word not in stopwords])

tokens = list(set(list(itertools.chain(*processed_sents))))
n_tokens = len(tokens)
print('Number of Sentences:', len(sents))
print('Number of Tokens:', n_tokens)

def get_co_occurences(token, processed_sents, window_size):
    co_occurences = []
    for sent in processed_sents:
        for idx in (np.array(sent)==token).nonzero()[0]:
            co_occurences.append(sent[max(0, idx-window_size):min(idx+window_size+1, len(sent))])

    co_occurences = list(itertools.chain(*co_occurences))
    co_occurence_idxs = list(map(lambda x: token2int[x], co_occurences))
    co_occurence_dict = Counter(co_occurence_idxs)
    co_occurence_dict = dict(sorted(co_occurence_dict.items()))
    return co_occurence_dict

def get_co_occurence_matrix(tokens, processed_sents, window_size):
    co_occurence_matrix = torch.zeros((len(tokens), len(tokens)), dtype=torch.float32, device=device)
    for token in tokens:
        token_idx = token2int[token]
        co_occurence_dict = get_co_occurences(token, processed_sents, window_size)
        indices = torch.tensor(list(co_occurence_dict.keys()), device=device)
        values = torch.tensor(list(co_occurence_dict.values()), dtype=torch.float32, device=device)
        co_occurence_matrix[token_idx, indices] = values

    co_occurence_matrix.fill_diagonal_(0)
    return co_occurence_matrix

def f(X_wc, X_max, alpha):
    return torch.where(X_wc < X_max, (X_wc / X_max) ** alpha, torch.ones_like(X_wc))

class WordVectorModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordVectorModel, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_bias = nn.Embedding(vocab_size, 1)
        self.context_bias = nn.Embedding(vocab_size, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.word_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
        nn.init.zeros_(self.word_bias.weight)
        nn.init.zeros_(self.context_bias.weight)

    def forward(self, word_idx, context_idx):
        word_embed = self.word_embeddings(word_idx)
        context_embed = self.context_embeddings(context_idx)
        word_bias = self.word_bias(word_idx).squeeze()
        context_bias = self.context_bias(context_idx).squeeze()
        return word_embed, context_embed, word_bias, context_bias

def loss_fn(word_embed, context_embed, word_bias, context_bias, X_wc, X_max, alpha):
    f_val = f(X_wc, X_max, alpha)
    loss = f_val * (torch.sum(word_embed * context_embed, dim=1) + word_bias + context_bias - torch.log(1 + X_wc)) ** 2
    return torch.sum(loss)

def train_model(model, co_occurence_matrix, n_tokens, embedding_size, n_epochs, alpha, eps, delta):
    optimizer = optim.Adagrad(model.parameters(), lr=alpha, lr_decay=delta)
    X_max = torch.max(co_occurence_matrix)

    norm_grad_weights = []
    norm_grad_bias = []
    costs = []
    n_iter = 0
    cost = 1
    convergence = 1

    co_occurence_pairs = torch.nonzero(co_occurence_matrix, as_tuple=True)
    word_indices, context_indices = co_occurence_pairs

    while cost > eps and n_iter < n_epochs:
        model.train()
        optimizer.zero_grad()

        word_embed, context_embed, word_bias, context_bias = model(word_indices, context_indices)
        X_wc = co_occurence_matrix[word_indices, context_indices]
        cost = loss_fn(word_embed, context_embed, word_bias, context_bias, X_wc, X_max, alpha)

        cost.backward()
        optimizer.step()

        if n_iter % 200 == 0:
            print(f'Cost at {n_iter} iterations:', cost.item())

        costs.append(cost.item())
        n_iter += 1

    if cost <= eps:
        print(f'Converged in {n_iter} epochs..')
    else:
        print(f'Training complete with {n_epochs} epochs..')
    return model, costs

def plotting(costs, last_n_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(costs[-last_n_epochs:], c='k')
    plt.title('cost')
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.show()

def plotting_word_vectors(model, similar_words, tokens):
    word_embeddings = model.word_embeddings.weight.data.cpu().numpy()
    pca = PCA(n_components=2)

    words_to_plot = similar_words + ['court']
    indices = [token2int[word] for word in words_to_plot]

    reduced_embeddings = pca.fit_transform(word_embeddings[indices])
    explained_var = (100 * sum(pca.explained_variance_ratio_)).round(2)
    print(f'Variance explained by 2 components: {explained_var}%')

    fig, ax = plt.subplots(figsize=(20, 10))
    for word, x1, x2 in zip(words_to_plot, reduced_embeddings[:, 0], reduced_embeddings[:, 1]):
        ax.annotate(word, (x1, x2))

    x_pad = 0.5
    y_pad = 1.5
    x_axis_min = np.amin(reduced_embeddings, axis=0)[0] - x_pad
    x_axis_max = np.amax(reduced_embeddings, axis=0)[0] + x_pad
    y_axis_min = np.amin(reduced_embeddings, axis=1)[1] - y_pad
    y_axis_max = np.amax(reduced_embeddings, axis=1)[1] + y_pad

    plt.xlim(x_axis_min, x_axis_max)
    plt.ylim(y_axis_min, y_axis_max)
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.show()

if bool_train:
    token2int = dict(zip(tokens, range(len(tokens))))
    int2token = {v: k for k, v in token2int.items()}

    print('Building co-occurrence matrix..')
    co_occurence_matrix = torch.zeros((len(tokens), len(tokens)), dtype=torch.float32, device=device)
    co_occurence_matrix = get_co_occurence_matrix(tokens, processed_sents, window_size)
    print('Co-occurrence matrix shape:', co_occurence_matrix.shape)
    assert co_occurence_matrix.shape == (n_tokens, n_tokens)

    # co-occurrence matrix is symmetric
    assert torch.all(co_occurence_matrix.T == co_occurence_matrix)

    print('\nTraining word vectors..')
    model = WordVectorModel(n_tokens, embedding_size).to(device)
    model, costs = train_model(model, co_occurence_matrix, n_tokens, embedding_size, n_epochs, alpha, eps, delta)

    # saving weights
    torch.save(model.state_dict(), 'model.pt')

def find_similar_words(word_embeddings, token, topN):
    token_idx = token2int[token]
    cosine_similarities = torch.mm(word_embeddings, word_embeddings[token_idx].unsqueeze(1)).squeeze(1)
    closest_word_indices = torch.argsort(cosine_similarities, descending=True)[1:topN + 1].cpu().numpy()
    closest_words = [int2token[idx] for idx in closest_word_indices]
    # also return cosine similarity of those closest words
    closest_similarities = cosine_similarities[closest_word_indices].cpu().numpy()
    return closest_words, closest_similarities
    # return closest_words

# getting word embeddings
word_embeddings = model.word_embeddings.weight.data

token = 'court'
closest_words, similarities = find_similar_words(word_embeddings, token, topN)
print(f'Similar words to {token}:', closest_words)
print(f'Cosine similarities:', similarities)

# loading pre-trained weights
print('Loading weights..')
model = WordVectorModel(n_tokens, embedding_size).to(device)
loaded_weights = torch.load('model.pt')
model.load_state_dict(loaded_weights)
model.eval()

print('Plotting learning curves..')
last_n_epochs = 3300
plotting(costs, last_n_epochs)

word_embeddings = model.word_embeddings.weight.data

token = 'court'
closest_words = find_similar_words(word_embeddings, token, topN)
print(f'Similar words to {token}:', closest_words)

# plotting similar words
plotting_word_vectors(model, closest_words, tokens)

"""# using library"""

from glove import Corpus, Glove

nltk.download('brown')
nltk.download('stopwords')

# Parameters
n_sents = 100  # number of sentences to consider
embedding_size = 20  # embedding dimension
n_epochs = 500  # number of training epochs
window_size = 5  # context window size
topN = 5  # top N similar words

# Processing sentences
brown = nltk.corpus.brown
sents = brown.sents()

print('Processing sentences..\n')
processed_sents = []
for sent in sents:
    processed_sents.append([word.lower() for word in sent if word.isalnum() and word not in stopwords])

tokens = list(set(itertools.chain(*processed_sents)))
n_tokens = len(tokens)
print('Number of Sentences:', len(sents))
print('Number of Tokens:', n_tokens)

# Building the GloVe corpus
corpus = Corpus()
corpus.fit(processed_sents, window=window_size)

# Training the GloVe model
glove = Glove(no_components=embedding_size, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=n_epochs, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# Function to find similar words
def find_similar_words(glove_model, token, topN):
    if token not in glove_model.dictionary:
        return []
    word_idx = glove_model.dictionary[token]
    word_vector = glove_model.word_vectors[word_idx]
    similarities = np.dot(glove_model.word_vectors, word_vector)
    most_similar = np.argsort(similarities)[::-1][1:topN+1]
    return [glove_model.inverse_dictionary[idx] for idx in most_similar]

# Finding similar words
token = 'court'
closest_words = find_similar_words(glove, token, topN)
print(f'Similar words to {token}:', closest_words)

# Plotting similar words
def plotting_word_vectors(glove_model, similar_words, token):
    words_to_plot = similar_words + [token]
    word_vectors = [glove_model.word_vectors[glove_model.dictionary[word]] for word in words_to_plot]

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(word_vectors)
    explained_var = (100 * sum(pca.explained_variance_ratio_)).round(2)
    print(f'Variance explained by 2 components: {explained_var}%')

    fig, ax = plt.subplots(figsize=(10, 5))
    for word, (x, y) in zip(words_to_plot, reduced_embeddings):
        ax.annotate(word, (x, y))
        plt.scatter(x, y)

    plt.title(f"2D PCA plot of the word '{token}' and its top {topN} similar words")
    plt.show()

# Plotting the similar words
plotting_word_vectors(glove, closest_words, token)
