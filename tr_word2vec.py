from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

def tsne(model):

    # t-SNE Visualization:

    vocab = list(model.wv.key_to_index)
    X = model.wv[vocab]
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
        ax.annotate(word, pos)

def analogy(model, worda, wordb, wordc):
    result = model.most_similar(negative=[worda],
                                positive=[wordb, wordc])

    return result[0][0]

def main():

    # Read the corpus:

    sentences = []
    fp = open('MT TR Corpus/Turkish Corpus.tr', 'r', encoding='utf-8')
    line = fp.readline()
    sentences.append(line)
    while line:
        line = fp.readline()
        sentences.append(line)

    # Input to Gensim Word2Vec implementation should be an iterable of sentences,
    # each consist of tokens seperated. Format should be 'utf-8'

    sentences = [s.split() for s in sentences]
    print(len(sentences))

    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=5)
    model.save("model/size100window5.model")

    # Test:

    model = Word2Vec.load("model/size100window5.model")
    word_vectors = model.wv

    print('numpy vector:', model.wv['çok'])  # get numpy vector of a word
    print('similars:', model.wv.most_similar('git', topn=5))

    # Evaluate Analogies

    test1 = analogy(word_vectors, 'erkek', 'baba', 'kadın')
    print(test1)
    test1 = analogy(word_vectors, 'sen', 'siz', 'ben')
    print(test1)
    test1 = analogy(word_vectors, 'bir', 'birinci', 'üç')
    print(test1)

    # Evaluate Similarity

if __name__ == '__main__':
    main()