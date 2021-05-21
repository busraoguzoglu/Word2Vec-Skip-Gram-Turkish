from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Ref: https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    count = 300
    for word in model.wv.vocab:
        if count > 0:
            tokens.append(model[word])
            labels.append(word)
            count -= 1

    print('tsne model will be created')
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

def main():

    print('This is the script for graphical evaluation')

    # Load Model:
    model1 = Word2Vec.load("models/model1.model")  # Vector:50 Window:3

    # tsne Plot:
    tsne_plot(model1)

if __name__ == '__main__':
    main()