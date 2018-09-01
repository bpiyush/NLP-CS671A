import re
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier

# f is the raw data file containing chapters 1-12 of the text
f = open("./FullDataset.txt")
raw = f.read()

# g is the file containing tagged sentences of data of file f
g = open("./taggedFullDataset.txt")
trainStr = g.read()
trainOutput = list(trainStr)

#h is the file containing testing data
h = open("./testData.txt")
testData = h.read()

h2 = open("./taggedTestData.txt")
taggedTestData = h2.read()

winSize = 10
featureLen = 7
dataSizeP = 0 # for period to be found
dataSizeE = 0 # for exclamation mark
dataSizeQ = 0 # for question mark

i = 0
j = 0
k = 0
for x in trainStr:
	if x == '.':
		dataSizeP += 1
	if x == '?':
		dataSizeQ += 1
	if x == '!':
		dataSizeE += 1

xP = np.zeros((dataSizeP, featureLen))
yP = np.zeros((dataSizeP, 1))
xE = np.zeros((dataSizeE, featureLen))

yE = np.zeros((dataSizeE, 1))
xQ = np.zeros((dataSizeQ, featureLen))
yQ = np.zeros((dataSizeQ, 1))

for idx, x in enumerate(trainStr):
	if x == '.':
		if trainStr[idx+1] == "<":
			yP[i] = 1
		elif trainStr[idx+1]=="'" and trainStr[idx+2] == '<':
			yP[i] = 1
		i += 1
	if x == '?':
		if trainStr[idx+1] == "<":
			yQ[j] = 1
		elif trainStr[idx+1]=="'" and trainStr[idx+2] == '<':
			yQ[j] = 1
		j += 1
	if x == '!':
		if trainStr[idx+1] == "<":
			yE[k] = 1
		elif trainStr[idx+1]=="'" and trainStr[idx+2] == '<':
			yE[k] = 1
		k += 1

def char_value(char):
	if 'A'<=char<='Z':
		return 2
	if 'a'<=char<='z':
		return -1
	if char==' ':
		return 3
	if char=="'":
		return 4
	if char=='\n':
		return 5
	return 0

def get_input(Str, idx, z, t):
	z[t][0] = char_value(Str[idx-1])
	z[t][1] = char_value(Str[idx-2])
	for k in range(3):
		z[t][k+2] = (char_value(Str[idx+k+2])+char_value(Str[idx-k-2]))/2
	z[t][5] = char_value(Str[idx+1])
	z[t][6] = char_value(Str[idx+2])


def input_map(raw):
	i=0
	j=0
	k=0
	b = 0
	for x in raw:
		if x in ["."]:
			get_input(raw, b, xP, i)
			i += 1
		if x in ["!"]:
			get_input(raw, b, xE, j)
			j += 1
		if x in ["?"]:
			get_input(raw, b, xQ, k)
			k += 1
		b += 1	

input_map(raw)


#Training phase-------------
"""
clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf1.fit(xP, yP)
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 4), random_state=1)
clf2.fit(xE, yE)
clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 4), random_state=1)
clf3.fit(xQ, yQ)
"""
clf1 = svm.SVC()
clf1.fit(xP, yP)
clf2 = svm.SVC()
clf2.fit(xE, yE)
clf3 = svm.SVC()
clf3.fit(xQ, yQ)
#---------------------------

# The following code is used for testing on Chapters 13,14 and 15 of the given text
def compute_data_size(string, c):
	count = 0
	for a in range(len(string)):
		if string[a] == c:
			count += 1
	return count

t1 = compute_data_size(testData, ".")
X_period = np.zeros((t1, featureLen))
Y_period = np.zeros((t1, 1))

t2 = compute_data_size(testData, "!")
X_em = np.zeros((t2, featureLen))
Y_em = np.zeros((t2, 1))

t3 = compute_data_size(testData, "?")
X_qm = np.zeros((t3, featureLen))
Y_qm = np.zeros((t3, 1))

def test_input_map(string, X, c):
	temp = 0
	for idx in range(len(string)):
		if string[idx] == c:
			get_input(string, idx, X, temp)
			temp += 1

test_input_map(testData, X_period, ".")
#print(X_period)
test_input_map(testData, X_em, "!")
#print(X_em)
test_input_map(testData, X_qm, "?")
#print(X_qm)

def compute_true_ouput(string, Y, c):
	it = 0
	for idx in range(len(string)):
		if string[idx] == c:
			if string[idx+1] == "<" or (string[idx+1]=="'" and string[idx+2] == '<'):
				#print(it, idx)
				Y[it] = 1
			it += 1
	pass

compute_true_ouput(taggedTestData, Y_period, ".")
#print(Y_period)

Y_period_predicted = clf1.predict(X_period)
#print(Y_period_predicted)

compute_true_ouput(taggedTestData, Y_em, "!")
#print(Y_em)

Y_em_predicted = clf2.predict(X_em)
#print(Y_em_predicted)

compute_true_ouput(taggedTestData, Y_qm, "?")
#print(Y_qm)

Y_qm_predicted = clf3.predict(X_qm)
#print(Y_qm_predicted)

def compute_accuracy(Y, Y_predicted):
	count2 = 0
	for x in range(len(Y)):
		if Y[x] != Y_predicted[x]:
			count2 += 1
	
	accuracy = 100 - (count2/len(Y))*100
	return accuracy
	pass

accuracy_period = compute_accuracy(Y_period, Y_period_predicted)
print("Accuracy for Period:", accuracy_period)

accuracy_em = compute_accuracy(Y_em, Y_em_predicted)
print("Accuracy for Exclamation mark:", accuracy_em)

accuracy_qm = compute_accuracy(Y_qm, Y_qm_predicted)
print("Accuracy for Question mark:", accuracy_qm)
