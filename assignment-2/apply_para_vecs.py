import gensim
import numpy as np
from gensim.models import Doc2Vec
import time
import tqdm
import os
import re

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

print("Loading model of DM para2vec:")
model = Doc2Vec.load('trained_models/para_vectors_dm.d2v')

# Construct the training arrays and labels
train_arrays = np.zeros((25000, 300))
train_labels = np.zeros(25000)
print("Creating training arrays and labels:")
for i in tqdm.trange(12500):
	train_arrays[i] = model['train_pos'+str(i)]
	train_labels[i] = 1
	train_arrays[i + 12500] = model['train_neg'+str(i+12500)]
	train_labels[i + 12500] = 0

# Construct the test arrays and labels
test_arrays = np.zeros((25000, 300))
test_labels = np.zeros(25000)
print("Creating test arrays and labels:")
for i in tqdm.trange(12500):
	test_arrays[i] = model['test_pos'+str(i+25000)]
	test_labels[i] = 1
	test_arrays[i + 12500] = model['test_neg'+str(i+37500)]
	test_labels[i + 12500] = 0


print("Training a Naive Bayes classifier:----")
gnb = GaussianNB()
gnb.fit(train_arrays, train_labels)
gnb_predicted = gnb.predict(test_arrays)
print("The accuracy of the model using Naive Bayes classifier and para2vec_dbow representation is: ", accuracy_score(test_labels, gnb_predicted)*100,"%")

print("Training a Logistic classifier:----")
LogReg = LogisticRegression()
LogReg.fit(train_arrays, train_labels)
LogReg_predicted = LogReg.predict(test_arrays)
print("The accuracy of the model using Logistic Regression and para2vec_dbow representation is: ", accuracy_score(test_labels, LogReg_predicted)*100,"%")

print("Training a SVM:----")
svm_model = svm.SVC()
svm_model.fit(train_arrays, train_labels)
svm_predicted = svm_model.predict(test_arrays)
print("The accuracy of the model using SVM  and para2vec_dbow representation is: ", accuracy_score(test_labels, svm_predicted)*100,"%")

print("Training a Neural Network:----")
mlp  = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
mlp.fit(train_arrays, train_labels)
mlp_predicted = mlp.predict(test_arrays)
print("The accuracy of the model using Neural Nets (MLP) and para2vec_dbow representation is: ", accuracy_score(test_labels, mlp_predicted)*100,"%")