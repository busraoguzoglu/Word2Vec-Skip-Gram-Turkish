# Word2Vec-Skip-Gram-Turkish

Used Gensim library.

Implemented to evaluate the Skip-gram algortithm using Gensim library on Turkish corpora. 

Dataset should be in same directory.

Input to Gensim Word2Vec implementation should be an iterable of sentences, each consist of tokens seperated. It can be in any language but format should be 'utf-8'.

This is made for analysis purposes, for the midterm exam of CMPE 58T Advanced NLP course on Bogazici University.

Example result 10 similar words to 'ben':

[('Ben', 0.8286421298980713), ('sen', 0.8155726790428162), ('siz', 0.8101901412010193), ('kendim', 0.8009457588195801), ('ikimiz', 0.7574027180671692), ('kendime', 0.7567036747932434), ('biz', 0.7440930008888245), ("Jenny'yi", 0.7342860698699951), ('oradaydım', 0.7242607474327087), ('benim', 0.7222325205802917)]

Analogy results for üç,üçüncü,iki:

[('ikinci', 0.8140236735343933), ('dördüncü', 0.7599884271621704), ('birinci', 0.758003294467926)]
