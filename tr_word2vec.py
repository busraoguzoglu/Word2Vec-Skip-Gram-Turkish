from gensim.models import Word2Vec
from time import time
import multiprocessing

# https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

def main():

    # Read the corpus:
    sentences = []
    fp = open('data/Turkish Corpus.tr', 'r', encoding='utf-8')
    line = fp.readline()
    sentences.append(line)
    while line:
        line = fp.readline()
        sentences.append(line)

    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer

    # Input to Gensim Word2Vec implementation should be an iterable of sentences,
    # each consist of tokens seperated. Format should be 'utf-8'
    sentences = [s.split() for s in sentences]
    print("Sentence splitting is finished, length is:", len(sentences))

    # Define Word2Vec Model
    model = Word2Vec(window=5, size=200, min_count=1, workers=cores-1)

    # Build Vocab
    t = time()
    model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    # Train Model
    t = time()
    model.train(sentences, total_examples=model.corpus_count, epochs=15, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    # Save Trained Model
    model.save("word2vec_size200_window5.model")

if __name__ == '__main__':
    main()