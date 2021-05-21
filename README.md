# Word2Vec-Skip-Gram-Turkish

Used Gensim library.

Implemented to evaluate the Skip-gram algortithm using Gensim library on Turkish corpora. 

Dataset should be in same directory.

Input to Gensim Word2Vec implementation should be an iterable of sentences, each consist of tokens seperated. It can be in any language but format should be 'utf-8'. It is better to seperate punctuation as well.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

This is made for analysis purposes, for the midterm exam of CMPE 58T Advanced NLP course on Bogazici University. Analysis is done using analogy and similarity pairs which were annotated by hand.

Overview of the implemented evaluation methods:

To evaluate similarities, a function was written, which checks similarity score for a given word pair using the model, and compare it with the scores in the file. All entries in the file were iterated, similarity scores were acquired using the models, and absolute difference of the scores (from the model and from the file) was calculated for each of the entries. These values were summed up and returned as total 'loss' in terms of score difference between the model and the file. Then these values were compared for different models. One of the things to note is that the similarity calculation function from the Gensim implementation returns values between 0 to 1, and in our file we had scores between 0 to 10. To be able to make a comparison, the scores in the file were normalized to the range 0 to 1. Another note is that not all the words in the files exist in the vocabulary. For such cases, those entries were ignored and were not included in the calculation.

To evaluate and compare the models using the analogies, a function was written which gets the analogy score using the model for given three words (read from file) and returns most 10 highest similarly scored words compared to the result. Then, if the forth word in the file is one of these 10 words, one point is given. It was checked and compared for all the entries in file (the ones in the vocabulary of the model, others were ignored) and final results (correctly found analogy count) were compared for different models. Comparison was made between the forth word in the file and 10 most similar words to analogy result because most of the time, the first found word is not the word in the file, and all models scored really bad in this setting. To get a more solid comparison, after checking different numbers, 10 was found to be a probable good setting to make this comparison.

Results regarding to best window size and vector size may change according to the vocabulary size of the used corpus.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

Example result 10 similar words to 'ben':

[('Ben', 0.8286421298980713), ('sen', 0.8155726790428162), ('siz', 0.8101901412010193), ('kendim', 0.8009457588195801), ('ikimiz', 0.7574027180671692), ('kendime', 0.7567036747932434), ('biz', 0.7440930008888245), ("Jenny'yi", 0.7342860698699951), ('oradaydım', 0.7242607474327087), ('benim', 0.7222325205802917)]

Analogy results for üç,üçüncü,iki:

[('ikinci', 0.8140236735343933), ('dördüncü', 0.7599884271621704), ('birinci', 0.758003294467926)]
