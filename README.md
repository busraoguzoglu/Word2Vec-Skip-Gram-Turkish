# Word2Vec-Skip-Gram-Turkish

Used Gensim library.

Implemented to evaluate the Skip-gram algortithm using Gensim library on Turkish corpora. 

Dataset should be in same directory.

Input to Gensim Word2Vec implementation should be an iterable of sentences, each consist of tokens seperated. It can be in any language but format should be 'utf-8'.

This is made for analysis purposes, for the midterm exam of CMPE 58T Advanced NLP course on Bogazici University.

Example result 10 similar words to 'sen':

similars: [('siz', 0.9491598606109619), ('ben', 0.9353943467140198), ('Sen', 0.9059412479400635), ('seni', 0.8968837857246399), ('seninle', 0.88191819190979), ('efendim,', 0.8742805123329163), ('evet,', 0.8655470013618469), ('sevgili', 0.8652328252792358), ('sanırım', 0.8643956184387207), ('Siz', 0.8619706630706787)]

