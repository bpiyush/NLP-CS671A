import gensim
import numpy as np
from gensim.models import Doc2Vec

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score

model = Doc2Vec.load('./imdb_train_test.d2v')

# Construct the training arrays and labels
train_arrays = np.zeros((25000, 300))
train_labels = np.zeros(25000)
for i in range(12500):
	train_arrays[i] = model['train_pos' + str(i)]
	train_arrays[12500 + i] = model['train_neg' + str(i + 12500)]
	train_labels[i] = 1
	train_labels[12500 + i] = 0

# Construct the test arrays and labels
test_arrays = np.zeros((25000, 300))
test_labels = np.zeros(25000)
for j in range(12500):
	i = j + 25000
	test_arrays[j] = model['test_pos' + str(i)]
	test_arrays[12500 + j] = model['test_neg' + str(i + 12500)]
	test_arrays[j] = 1
	test_arrays[12500 + j] = 0


# Gaussian Naive Bayes Model
gnb = GaussianNB()
# Train the Naive Bayes classifier
gnb.fit(train_arrays, train_labels)

# Predict the output for test array
gnb_test_predicted = gnb.predict(test_arrays)

print("The accuracy of the model using Naive Bayes is: ", accuracy_score(test_labels, gnb_test_predicted)*100,"%")

LogReg = LogisticRegression()
LogReg.fit(train_arrays, train_labels)
LogReg_predicted = LogReg.predict(test_arrays)

print("The accuracy of the model using Logistic Regression is: ", LogReg.score(test_labels, LogReg_predicted)*100,"%")

svm_model = svm.SVC()
svm_model.fit(train_arrays, train_labels)
svm_predicted = svm_model.predict(test_arrays)

print(svm_predicted)
print("The accuracy of the model using SVM is: ", metrics.accuracy_score(test_labels, svm_predicted)*100,"%")