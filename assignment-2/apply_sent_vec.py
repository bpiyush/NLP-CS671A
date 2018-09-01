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

print("Loading model of sentence2vec:")
model = Doc2Vec.load('trained_models/sent_vectors_dm.s2v')

folders = ['aclImdb/train/pos', 'aclImdb/train/neg', 'aclImdb/test/pos', 'aclImdb/test/neg']

def sort_custom(files):
    l = len(files)
    file_ids = np.zeros(l)
    for i, x in enumerate(files):
        pattern = re.compile(r"\d{1,5}_")
        match = pattern.findall(x)
        match[0] = match[0][:-1]
        t = int(match[0])
        file_ids[i] = t
    X = files
    Y = file_ids
    Z = [x for _,x in sorted(zip(Y,X))]
    return Z
    pass

def get_doc_list(folder_name):
    doc_list = [] # Has the list of all documents (text data)
    # file_list has the names of all the files in the folder_name
    file_list = [folder_name+'/'+name for name in os.listdir(folder_name) if name.endswith('txt')]
    file_list = sort_custom(file_list)
    for file in file_list:
        st = open(file,'r').read()
        doc_list.append(st)
    print ('Found %s documents under the dir %s .....'%(len(file_list),folder_name))
    return doc_list

data = []
for j in tqdm.trange(4):
	t = get_doc_list(folders[j])
	for s in range(len(t)):
		data.append(t[s])

def split_doc_into_sent(doc):
    pattern = re.compile(r"([^A-Z]|I)[^A-Z\.](\.|\.'|\?|\?'|!|!'|-')\s+[A-Z'*][^\.]")
    match = pattern.finditer(doc)
    split_idx = []
    split_idx.append(0)
    for x in match:
        split_idx.append(x.span()[0]+3)
    parts = [doc[i:j] for i,j in zip(split_idx, split_idx[1:]+[None])]
    return parts
    pass

sentence_id = []
for z in data:
    y = split_doc_into_sent(z)
    if len(sentence_id) >= 1:
        h = sentence_id[len(sentence_id)-1]
    else:
        h = 0
    sentence_id.append(h + len(y)) # Number of sentences in given document

# Construct the training arrays and labels
train_arrays = np.zeros((25000, 300))
train_labels = np.zeros(25000)
print("Creating training arrays and labels:")
for i in tqdm.trange(12500):
	if i > 0:
		for z in range(sentence_id[i-1], sentence_id[i]):
			train_arrays[i] += model['train_pos' + "doc_id" + str(i) + "sent_id" + str(z)]
			# train_arrays[i + 12500] += model['train_neg' + "doc_id" + str(i + 12500) + "sent_id" + str(z)]
		train_arrays[i] = (1/(sentence_id[i]-sentence_id[i-1]))*train_arrays[i]
	else:
		for z in range(0, sentence_id[i]):
			train_arrays[i] += model['train_pos' + "doc_id" + str(i) + "sent_id" + str(z)]
			# train_arrays[i + 12500] += model['train_neg' + "doc_id" + str(i + 12500) + "sent_id" + str(z)]
		train_arrays[i] = (1/(sentence_id[i]))*train_arrays[i]
	train_labels[i] = 1
	# train_labels[12500 + i] = 0

for i in tqdm.trange(12500, 25000):
	for z in range(sentence_id[i-1], sentence_id[i]):
		# train_arrays[i] += model['train_pos' + "doc_id" + str(i) + "sent_id" + str(z)]
		train_arrays[i] += model['train_neg' + "doc_id" + str(i) + "sent_id" + str(z)]
	train_arrays[i] = (1/(sentence_id[i]-sentence_id[i-1]))*train_arrays[i]
	train_labels[i] = 0

# Construct the test arrays and labels
test_arrays = np.zeros((25000, 300))
test_labels = np.zeros(25000)
print("Creating test arrays and labels:")
for j in tqdm.trange(12500):
	for z in range(sentence_id[j-1 + 25000], sentence_id[j + 25000]):
		test_arrays[j] += model['test_pos' + "doc_id" + str(j + 25000) + "sent_id" + str(z)]
		# test_arrays[j + 12500] += model['test_neg' + "doc_id" + str(i + 12500) + "sent_id" + str(z)]
	test_arrays[j] = (1/(sentence_id[j + 25000]-sentence_id[j-1 + 25000]))*test_arrays[j]
	test_labels[j] = 1

for j in tqdm.trange(12500):
	for z in range(sentence_id[j-1 + 37500], sentence_id[j + 37500]):
		# test_arrays[j] += model['test_pos' + "doc_id" + str(j + 37500) + "sent_id" + str(z)]
		test_arrays[j + 12500] += model['test_neg' + "doc_id" + str(j + 37500) + "sent_id" + str(z)]
	test_arrays[j + 12500] = (1/(sentence_id[j + 37500]-sentence_id[j-1 + 37500]))*test_arrays[j + 12500]
	test_labels[j + 12500] = 0

order = np.arange(train_labels.shape[0])
np.random.shuffle(order)
idx = np.argsort(order)
train_arrays = train_arrays[idx,:]
train_labels = train_labels[idx]

"""
print("Training a Naive Bayes classifier:----")
gnb = GaussianNB()
gnb.fit(train_arrays, train_labels)
gnb_predicted = gnb.predict(test_arrays)
print("The accuracy of the model using Naive Bayes classifier and average of sentence vectors representation is: ", accuracy_score(test_labels, gnb_predicted)*100,"%")

print("Training a Logistic classifier:----")
LogReg = LogisticRegression()
LogReg.fit(train_arrays, train_labels)
LogReg_predicted = LogReg.predict(test_arrays)
print("The accuracy of the model using Logistic Regression and average of sentence vectors representation is: ", accuracy_score(test_labels, LogReg_predicted)*100,"%")

print("Training a SVM:----")
svm_model = svm.SVC()
svm_model.fit(train_arrays, train_labels)
svm_predicted = svm_model.predict(test_arrays)
print("The accuracy of the model using SVM  and average of sentence vectors representation is: ", accuracy_score(test_labels, svm_predicted)*100,"%")"""

print("Training a Neural Network:----")
mlp  = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(8, 5), random_state=1)
mlp.fit(train_arrays, train_labels)
mlp_predicted = mlp.predict(test_arrays)
print("The accuracy of the model using Neural Nets (MLP) and average of sentence vectors representation is: ", accuracy_score(test_labels, mlp_predicted)*100,"%")