from gensim.models import Word2Vec
from time import time
import multiprocessing
import nltk

#Ref: https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

def main():

    # Read the corpus:
    sentences = []
    fp = open('data/Turkish Corpus.tr', 'r', encoding='utf-8')
    line = fp.readline()
    sentences.append(line)
    while line:
        line = fp.readline()
        sentences.append(line)

    cores = multiprocessing.cpu_count()  # Count the number of cores in the computer

    # Input to Gensim Word2Vec implementation should be an iterable of sentences,
    # each consist of tokens seperated. Format should be 'utf-8'
    nltk.download('punkt')
    sentences = [nltk.word_tokenize(s) for s in sentences]
    print("Sentence splitting is finished, length is:", len(sentences))

    # Define Word2Vec Model
    # sg = 1 -> Uses Skip Gram Implementation
    # sg = 0 -> Uses CBOW Implementation
    model = Word2Vec(window=5, size=150, min_count=1, workers=cores-1, sg=1)

    # Build Vocab
    t = time()
    model.build_vocab(sentences, progress_per=10000)
    print('Building vocabulary is finihed in {} mins'.format(round((time() - t) / 60, 2)))

    # Train Model
    t = time()
    model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print('Model training is finished in {} mins'.format(round((time() - t) / 60, 2)))

    # Save Trained Model
    model.save("models/model6.model")

if __name__ == '__main__':
    main()