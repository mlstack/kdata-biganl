#!/usr/bin/env /c/Apps/Anaconda3/python
"""
# 
# [quick learn, word2vec, Unsupervised]
import codecs
import gensim, logging
from gensim.models import word2vec, Word2Vec
from pprint import pprint
sentences = word2vec.Text8Corpus('text8')
model = word2vec.Word2Vec(sentences, size=200, workers=12, min_count=3, sg=0, window=8, iter=15, sample=1e-4, negative=25)
word_similarity = model.similarity('king','queen')
pprint(word_similarity)
word_matching = model.most_similar(positive=['king','queen'],negative=['man'],topn=100)
for i in range(len(word_matching)):
	print(i, word_matching[i])
"""
# print(__doc__)
import codecs
import gensim, logging

# conda install -c anaconda gensim 
from gensim.models import word2vec, Word2Vec
from pprint import pprint
from itertools import chain
from glob import glob
lines = set(chain.from_iterable(codecs.open(f, 'r',encoding="utf-8") for f in glob('machine_learning.txt')))
lines = [line.lower() for line in lines]
with codecs.open('machine_learning-lower.txt', 'w',encoding="utf-8") as out:
	out.writelines(sorted(lines))

sentences = word2vec.Text8Corpus('machine_learning-lower.txt')

model = word2vec.Word2Vec(
		sentences
	, 	size=200
	, 	workers=12
	, 	min_count=3
	, 	sg=0
	, 	window=5
	, 	iter=100
	, 	sample=1e-4
	, 	negative=25
	)

word1 = 'machine'
word2 = 'learning'
word_similarity = model.similarity(word1,word2)
negative_word = 'metric'
word_matching = model.most_similar(positive=[word1,word2],negative=[negative_word],topn=20)
print(word1,":",word2,"=",negative_word,": ???" )
for i in range(len(word_matching)):
	print(i, word_matching[i])

word1 = 'learning'
word2 = 'data'
word_similarity = model.similarity(word1,word2)
print("Similarity[",word1,",",word2,"] ", word_similarity)




