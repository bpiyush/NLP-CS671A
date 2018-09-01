import re
import numpy as np
import gensim.models as gsm
import os
import stop_words
import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

folders = ['aclImdb/train/pos', 'aclImdb/train/neg', 'aclImdb/test/pos', 'aclImdb/test/neg']

bowFile = open("./aclImdb/train/labeledBow.feat")
bowRaw = bowFile.read()

testBowFile = open("./aclImdb/test/labeledBow.feat")
testBowRaw = testBowFile.read()

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
vocabLen = len(corpusVocab)

print( "Number of words in the Vocabulary:",len(corpusVocab))

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

bbowVectorize = CountVectorizer(binary = True, vocabulary = corpusVocab)
tfidfVectorize = TfidfVectorizer(vocabulary = corpusVocab)
normVectorize = CountVectorizer(vocabulary = corpusVocab)

X_counts = bbowVectorize.fit_transform(data)
X_tfidf = tfidfVectorize.fit_transform(data)
X_tf = normVectorize.fit_transform(data)
"""
print(X_counts.shape)
print(X_tf.shape)
print(X_tfidf.shape)"""

X_labels = np.zeros(25000)
for j in range(25000):
    if j<12500:
        X_labels[j] = 1

def get_bbow():
    return X_counts
    pass
def get_tfidf():
    return X_tfidf
    pass
train_data_count = X_counts[0:25000, :]
test_data_count = X_counts[25000:50000, :]
train_data_tf = X_tf[0:25000, :]
test_data_tf = X_tf[25000:50000, :]
train_data_tfidf = X_tfidf[0:25000, :]
test_data_tfidf = X_tfidf[25000:50000, :]

def training_models(rep, train_arrays, test_arrays, train_labels, test_labels):
    """print("Training a Naive Bayes classifier:----")
                gnb = MultinomialNB()
                gnb.fit(train_arrays, train_labels)
                gnb_predicted = gnb.predict(test_arrays)
                print("The accuracy of the model using Naive Bayes classifier and " + rep + " representation is: ", accuracy_score(test_labels, gnb_predicted)*100,"%")
            
                print("Training a Logistic classifier:----")
                LogReg = LogisticRegression()
                LogReg.fit(train_arrays, train_labels)
                LogReg_predicted = LogReg.predict(test_arrays)
                print("The accuracy of the model using Logistic Regression and " + rep + " representation is: ", accuracy_score(test_labels, LogReg_predicted)*100,"%")
            
                print("Training a SVM:----")
                svm_model = svm.SVC()
                svm_model.fit(train_arrays, train_labels)
                svm_predicted = svm_model.predict(test_arrays)
                print("The accuracy of the model using SVM  and " + rep + " representation is: ", accuracy_score(test_labels, svm_predicted)*100,"%")"""

    print("Training a Neural Network:----")
    mlp  = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(10, 2), random_state=1)
    mlp.fit(train_arrays, train_labels)
    mlp_predicted = mlp.predict(test_arrays)
    print("The accuracy of the model using Neural Nets (MLP) and " + rep + " representation is: ", accuracy_score(test_labels, mlp_predicted)*100,"%")

    pass
"""
training_models('BBoW',train_data_count, test_data_count, X_labels, X_labels)
training_models('nTF',train_data_tf, test_data_tf, X_labels, X_labels)
training_models('TFIDF',train_data_tfidf, test_data_tfidf, X_labels, X_labels)"""