from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

def evaluate_analogy(model):
    # Read the Evaluation Script:
    analogy_entries = []
    fp = open('data/Analogy.txt', 'r', encoding='utf-8')
    line = fp.readline()
    analogy_entries.append(line)
    while line:
        line = fp.readline()
        analogy_entries.append(line)

def evaluate_similarity(model):
    # Read the Evaluation Script:
    similarity_entries = []
    fp = open('data/Word Similarity.txt', 'r', encoding='utf-8')
    line = fp.readline()
    similarity_entries.append(line)
    while line:
        line = fp.readline()
        similarity_entries.append(line)

    difference_value = 0

    for i in range (len(similarity_entries)-1):
        entry = similarity_entries[i]
        splitted = entry.split(':')
        word1 = splitted[0]
        word2 = splitted[1].split('\t')[0]
        score = (float)(splitted[1].split('\t')[1])/10

        # Check if word1 and word2 is in vocab
        model_score = model.wv.similarity(word1, word2)
        difference_value += np.abs(score-model_score)

    return difference_value

def analogy(model, a, b, c):

    result = model.wv.most_similar(positive=[a, b], negative=[c], topn=3)
    return result[0][0]

def main():

    print('Hello this script is for evalution')

    # Test:
    model = Word2Vec.load("word2vec_size200_window5.model")

    # Most Similar:
    print('\n', model.wv.most_similar(positive=["ben"]))

    # Evaluate Analogies
    print('\n', model.wv.most_similar(positive=["kadın", "baba"], negative=["anne"], topn=3))
    print('\n', model.wv.most_similar(positive=["üçüncü", "iki"], negative=["üç"], topn=3))
    print('\n', model.wv.most_similar(positive=["siz", "ben"], negative=["sen"], topn=3))

    # Evaluate Similarity
    print('\n', model.wv.similarity("kız", "kadın"))

    evaluate_similarity(model)

    # Plotting:

if __name__ == '__main__':
    main()