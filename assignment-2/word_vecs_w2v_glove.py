import gensim.models as gsm
import numpy as np
import tqdm
import os
import temp8
from gensim.scripts.glove2word2vec import glove2word2vec
import re

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

glove_input_file = 'glove.6B/glove.6B.300d.txt'
word2vec_output_file = 'glove.6B.300d.txt.word2vec'

glove2word2vec(glove_input_file, word2vec_output_file)

vocabFile = open("./aclImdb/imdb.vocab")
vocab = vocabFile.read()

def convert_to_list(string):
	# Given a string, breaks up into substrings separated by a newline
	t = ''
	temp = []
	for x in string:
		if x not in ['\n']:
			t += x
		else:
			temp.append(t)
			t = ''
	return temp
	pass

corpusVocab = convert_to_list(vocab)

print("Loading models...")
# Load the Google word2vec model
w2v = gsm.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
# load the Stanford GloVe model
gloveModel = gsm.KeyedVectors.load_word2vec_format('glove.6B.300d.txt.word2vec', binary=False, limit = 500000)

batchSize = 25000
bow = temp8.get_bbow()
tfidf = temp8.get_tfidf()

def form_wv(datatype, label):
	if datatype == 'training':
		t_bbow = bow[0:25000,:]
		t_tfidf = tfidf[0:25000,:]
	else:
		t_bbow = bow[25000:50000,:]
		t_tfidf = tfidf[25000:50000,:]
	wordVecAvg = []
	weighted_wva = []
	gloveAvg = []
	weightedGloveAvg = []
	print("Constructing word vectors for "+ datatype + "ing with label "+ label)
	for doc in tqdm.trange(batchSize):
		docAvg = np.zeros(300)
		weightedDocAvg = np.zeros(300)
		gloveDocAvg = np.zeros(300)
		weightedGloveDocAvg = np.zeros(300)
		factor = np.sum(bbow[doc])
		for j,x in enumerate(t_bbow[doc]):
			if x != 0:
				if corpusVocab[j] in w2v.vocab:
					wv = w2v[corpusVocab[j]]	
					docAvg += wv
					weightedDocAvg += (t_tfidf[doc][j])*wv
				if corpusVocab[j] in gloveModel.vocab:
					gVec = gloveModel[corpusVocab[j]]
					gloveDocAvg += gVec
					weightedGloveDocAvg += (t_tfidf[doc][j])*gVec
		wordVecAvg.append(docAvg/factor)
		weighted_wva.append(weightedDocAvg/factor)
		gloveAvg.append(gloveDocAvg/factor)
		weightedGloveAvg.append(weightedGloveDocAvg/factor)
	return [wordVecAvg, weighted_wva, gloveAvg, weightedGloveAvg]
	pass

def training_models(wordVec, wordVecType):
	train_arrays = np.zeros((batchSize, 300))
	train_labels = np.zeros(batchSize)
	print("Constructing the training arrays and labels:")
	for j in tqdm.trange(batchSize):
		train_arrays[j] = wordVec[j]
		train_arrays[j+12500] = wordVec[j+12500]
		train_labels[j] = 1
		train_labels[j + 12500] = 0

	test_arrays = np.zeros((batchSize, 300))
	test_labels = np.zeros(batchSize)
	print("Constructing the test arrays and labels:")
	for j in tqdm.trange(batchSize):
		test_arrays[j] = wordVec[j + 25000]
		test_labels[j] = 1
		test_arrays[j + 12500] = wordVec[j+37500]
		test_labels[j + 12500] = 0
	print("Training a Naive Bayes classifier:----")
	gnb = GaussianNB()
	gnb.fit(train_arrays, train_labels)
	gnb_predicted = gnb.predict(test_arrays)
	print("The accuracy of the model using Naive Bayes classifier and" + wordVecType + "representation is: ", accuracy_score(test_labels, gnb_predicted)*100,"%")

	print("Training a Logistic classifier:----")
	LogReg = LogisticRegression()
	LogReg.fit(train_arrays, train_labels)
	LogReg_predicted = LogReg.predict(test_arrays)
	print("The accuracy of the model using Logistic Regression and" + wordVecType + "representation is: ", accuracy_score(test_labels, LogReg_predicted)*100,"%")

	print("Training a SVM:----")
	svm_model = svm.SVC()
	svm_model.fit(train_arrays, train_labels)
	svm_predicted = svm_model.predict(test_arrays)
	print("The accuracy of the model using SVM  and " + wordVecType + "representation is: ", accuracy_score(test_labels, svm_predicted)*100,"%")

	print("Training a Neural Network:----")
	mlp  = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
	mlp.fit(train_arrays, train_labels)
	mlp_predicted = mlp.predict(test_arrays)
	print("The accuracy of the model using Neural Nets (MLP) and " + wordVecType + "representation is: ", accuracy_score(test_labels, mlp_predicted)*100,"%")
	pass

training_models(wordVecAvg, 'wordVecAvg')
training_models(weighted_wva, 'weighted_wva')
training_models(gloveAvg, 'gloveAvg')
training_models(weightedGloveAvg, 'weightedGloveAvg')