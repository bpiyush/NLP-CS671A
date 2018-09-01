import re
import numpy as np
import os
import tqdm

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

folder = 'feature_vecs/'
repMethod = 'tfidf'

batchSize = 500

# Construct the training arrays and labels
train_arrays = np.zeros((1000, 89526))
train_labels = np.zeros(1000)
print("Loading training arrays and labels:")
train_pos = np.load(folder + 'train_pos_' + repMethod + '.npy')
train_neg = np.load(folder + 'train_neg_' + repMethod + '.npy')
for i in tqdm.trange(batchSize):
	train_arrays[i] = train_pos[i]
	train_labels[i] = 1
	train_arrays[i+batchSize] = train_neg[i]
	train_labels[i+batchSize] = 0

# Construct the test arrays and labels
test_arrays = np.zeros((1000, 89526))
test_labels = np.zeros(1000)
print("Loading test arrays and labels:")
test_pos = np.load(folder + 'test_pos_' + repMethod + '.npy')
test_neg = np.load(folder + 'test_neg_' + repMethod + '.npy')
for i in tqdm.trange(batchSize):
	test_arrays[i] = test_pos[i]
	test_labels[i] = 1
	test_arrays[i+batchSize] = test_neg[i]
	test_labels[i+batchSize] = 0

order = np.arange(train_labels.shape[0])
np.random.shuffle(order)
idx = np.argsort(order)
train_arrays = train_arrays[idx,:]
train_labels = test_labels[idx]

print("Training a Naive Bayes classifier:----")
gnb = GaussianNB()
gnb.fit(train_arrays, train_labels)
gnb_predicted = gnb.predict(test_arrays)
print("The accuracy of the model using Naive Bayes classifier and BBOW representation is: ", accuracy_score(test_labels, gnb_predicted)*100,"%")

print("Training a Logistic classifier:----")
LogReg = LogisticRegression()
LogReg.fit(train_arrays, train_labels)
LogReg_predicted = LogReg.predict(test_arrays)
print("The accuracy of the model using Logistic Regression and BBOW representation is: ", accuracy_score(test_labels, LogReg_predicted)*100,"%")

print("Training a SVM:----")
svm_model = svm.SVC()
svm_model.fit(train_arrays, train_labels)
svm_predicted = svm_model.predict(test_arrays)
print("The accuracy of the model using SVM  and BBOW representation is: ", accuracy_score(test_labels, svm_predicted)*100,"%")

print("Training a Neural Network:----")
mlp = clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(train_arrays, train_labels)
mlp_predicted = mlp.predict(test_arrays)
print("The accuracy of the model using Neural Nets (MLP) and BBOW representation is: ", accuracy_score(test_labels, mlp_predicted)*100,"%")
