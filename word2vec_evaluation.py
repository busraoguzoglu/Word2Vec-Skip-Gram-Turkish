from gensim.models import Word2Vec
import numpy as np

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

            # Comparison with Results:
            exists = False
            for i in range(10):
                if results[i][0] == word3:
                    exists = True

            # Point +1 if the target word is one of the 10 guessed words:
            if exists:
                found_count+=1

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
    model1 = Word2Vec.load("models/model1.model") # Vector:50 Window:3
    model2 = Word2Vec.load("models/model2.model") # Vector:100 Window:3
    model3 = Word2Vec.load("models/model3.model") # Vector:150 Window:3
    model4 = Word2Vec.load("models/model4.model") # Vector:50 Window:5
    model5 = Word2Vec.load("models/model5.model") # Vector:100 Window:5
    model6 = Word2Vec.load("models/model6.model") # Vector:150 Window:5

    # Most Similar:
    print('\n', model1.wv.most_similar(positive=["ben"]))

    # Evaluate Analogies
    print('\n', model1.wv.most_similar(positive=["kadın", "baba"], negative=["anne"], topn=3))
    print('\n', model1.wv.most_similar(positive=["üçüncü", "iki"], negative=["üç"], topn=3))
    print('\n', model1.wv.most_similar(positive=["siz", "ben"], negative=["sen"], topn=3))

    print('Total correctly found analogies for model1: ', evaluate_analogy(model1))
    print('Total correctly found analogies for model2: ', evaluate_analogy(model2))
    print('Total correctly found analogies for model3: ', evaluate_analogy(model3))
    print('Total correctly found analogies for model4: ', evaluate_analogy(model4))
    print('Total correctly found analogies for model5: ', evaluate_analogy(model5))
    print('Total correctly found analogies for model6: ', evaluate_analogy(model6))

    # Evaluate Similarity
    print('\n', model1.wv.similarity("kız", "kadın"))

    print('\nTotal difference for model1: ', evaluate_similarity(model1))
    print('Total difference for model2: ', evaluate_similarity(model2))
    print('Total difference for model3: ', evaluate_similarity(model3))
    print('Total difference for model4: ', evaluate_similarity(model4))
    print('Total difference for model5: ', evaluate_similarity(model5))
    print('Total difference for model6: ', evaluate_similarity(model6))

    # Plotting:

if __name__ == '__main__':
    main()