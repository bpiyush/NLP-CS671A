import numpy as np
import time
import tqdm
import os
import re

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

folder = 'feature_vecs/'
repMethod = 'weightedGloveAvg'

if repMethod == 'WVAvg':
	idx = 0
elif repMethod == 'weightedWVAvg':
	idx = 1
else:
	if repMethod == 'gloveAvg':
		idx = 2
	elif repMethod == 'weightedGloveAvg':
		idx = 3

batchSize = 500

print("Loading the trained pos/neg word_vecs:")
train_pos_vec = np.load(folder + 'train_pos_vec.npy')
train_neg_vec = np.load(folder + 'train_neg_vec.npy')

train_arrays = np.zeros((2*batchSize, 300))
train_labels = np.zeros(2*batchSize)
print("Constructing the training arrays and labels:")
for j in tqdm.trange(batchSize):
	train_arrays[j] = train_pos_vec[idx][j]
	train_labels[j] = 1
	train_arrays[j + batchSize] = train_neg_vec[idx][j]
	train_labels[j + batchSize] = 0

print("Loading the test pos/neg word_vecs:")
test_pos_vec = np.load(folder + 'test_pos_vec.npy')
test_neg_vec = np.load(folder + 'test_neg_vec.npy')

test_arrays = np.zeros((2*batchSize, 300))
test_labels = np.zeros(2*batchSize)
print("Constructing the test arrays and labels:")
for j in tqdm.trange(batchSize):
	test_arrays[j] = test_pos_vec[idx][j]
	test_labels[j] = 1
	test_arrays[j + batchSize] = test_neg_vec[idx][j]
	test_labels[j + batchSize] = 0
"""
order = np.arange(train_labels.shape[0])
np.random.shuffle(order)
idx = np.argsort(order)
train_arrays = train_arrays[idx,:]
train_labels = train_labels[idx]"""

print("Training a Naive Bayes classifier:----")
gnb = GaussianNB()
gnb.fit(train_arrays, train_labels)
gnb_predicted = gnb.predict(test_arrays)
print("The accuracy of the model using Naive Bayes classifier and word2vec representation is: ", accuracy_score(test_labels, gnb_predicted)*100,"%")

print("Training a Logistic classifier:----")
LogReg = LogisticRegression()
LogReg.fit(train_arrays, train_labels)
LogReg_predicted = LogReg.predict(test_arrays)
print("The accuracy of the model using Logistic Regression and word2vec representation is: ", accuracy_score(test_labels, LogReg_predicted)*100,"%")

print("Training a SVM:----")
svm_model = svm.SVC()
svm_model.fit(train_arrays, train_labels)
svm_predicted = svm_model.predict(test_arrays)
print("The accuracy of the model using SVM  and word2vec representation is: ", accuracy_score(test_labels, svm_predicted)*100,"%")

print("Training a Neural Network:----")
mlp  = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
mlp.fit(train_arrays, train_labels)
mlp_predicted = mlp.predict(test_arrays)
print("The accuracy of the model using Neural Nets (MLP) and word2vec representation is: ", accuracy_score(test_labels, mlp_predicted)*100,"%")
