from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

def analogy(model, a, b, c):

    result = model.wv.most_similar(positive=[a, b], negative=[c], topn=3)
    return result[0][0]

def main():

    print('Hello this script is for evalution')

    # Test:
    model = Word2Vec.load("word2vec.model")

    # Most Similar:
    print('\n', model.wv.most_similar(positive=["ben"]))

    # Evaluate Analogies
    print('\n', model.wv.most_similar(positive=["kadın", "baba"], negative=["anne"], topn=3))
    print('\n', model.wv.most_similar(positive=["üçüncü", "iki"], negative=["üç"], topn=3))
    print('\n', model.wv.most_similar(positive=["siz", "ben"], negative=["sen"], topn=3))

    # Evaluate Similarity
    print('\n', model.wv.similarity("anne", "baba"))

    # Plotting:

if __name__ == '__main__':
    main()