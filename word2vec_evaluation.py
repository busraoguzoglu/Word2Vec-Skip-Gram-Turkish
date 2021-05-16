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

    found_count = 0
    not_found_count = 0
    total_count = 0

    for index in range (len(analogy_entries)-1):
        entry = analogy_entries[index]
        seperated = entry.split(' ')
        word1 = seperated[0]
        word2 = seperated[1]
        word3 = seperated[2]
        word4 = seperated[3].strip()

        # Check if word1, word2, word3 and word4 is in vocab
        if word1 in model.wv.vocab and word2 in model.wv.vocab and word3 in model.wv.vocab and word4 in model.wv.vocab:
            results = analogy(model, word1, word4, word2)
            total_count += 1

            # Point +1 if the target word is one of the 10 guessed words:
            if results[0][0] == word3 or results[1][0] == word3 or results[2][0] == word3\
                    or results[3][0] == word3 or results[4][0] == word3\
                    or results[5][0] == word3 or results[6][0] == word3\
                    or results[7][0] == word3 or results[8][0] == word3\
                    or results[9][0] == word3:
                found_count+=1
            else:
                not_found_count +=1

    return found_count

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
        if word1 in model.wv.vocab and word2 in model.wv.vocab:
            model_score = model.wv.similarity(word1, word2)
            difference_value += np.abs(score-model_score)

    return difference_value

def analogy(model, a, b, c):

    result = model.wv.most_similar(positive=[a, b], negative=[c], topn=10)
    return result

def main():

    print('Hello this script is for evalution')

    # Test:
    model1 = Word2Vec.load("word2vec_size100_window3.model")
    model2 = Word2Vec.load("word2vec_size200_window3.model")
    model3 = Word2Vec.load("word2vec_size100_window5.model")
    model4 = Word2Vec.load("word2vec_size200_window5.model")

    # Most Similar:
    print('\n', model4.wv.most_similar(positive=["ben"]))

    # Evaluate Analogies
    print('\n', model4.wv.most_similar(positive=["kadın", "baba"], negative=["anne"], topn=3))
    print('\n', model4.wv.most_similar(positive=["üçüncü", "iki"], negative=["üç"], topn=3))
    print('\n', model4.wv.most_similar(positive=["siz", "ben"], negative=["sen"], topn=3))

    print('Total correctly found analogies for model1: ', evaluate_analogy(model1))
    print('Total correctly found analogies for model2: ', evaluate_analogy(model2))
    print('Total correctly found analogies for model3: ', evaluate_analogy(model3))
    print('Total correctly found analogies for model4: ', evaluate_analogy(model4))

    # Evaluate Similarity
    print('\n', model4.wv.similarity("kız", "kadın"))

    print('\nTotal difference for model1: ', evaluate_similarity(model1))
    print('Total difference for model2: ', evaluate_similarity(model2))
    print('Total difference for model3: ', evaluate_similarity(model3))
    print('Total difference for model4: ', evaluate_similarity(model4))

    # Plotting:

if __name__ == '__main__':
    main()