from gensim.models import Word2Vec

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

    #model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, workers=4)
    #model.save("word2vec.model")

    # Test:

    model = Word2Vec.load("word2vec.model")

    print('numpy vector:', model.wv['çok'])  # get numpy vector of a word
    print('similars:', model.wv.most_similar('sen', topn=10))

if __name__ == '__main__':
    main()