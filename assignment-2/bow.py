import re
import numpy as np
import tqdm
import os

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
corpusBow = convert_to_list(bowRaw)
testBow = convert_to_list(testBowRaw)
corpusLen = len(corpusBow)
vocabLen = len(corpusVocab)
testBowLen = len(testBow)

print( "Number of words in the Vocabulary:",len(corpusVocab))
print("Number of training documents in the corpus:", len(corpusBow))
print("Number of test documents in the corpus:", len(testBow))

batchSize = 500
print("Testing the code for batchSize = 1000")

bowVecSet = []
testBowVecSet = []
nBowVecSet = []
testnBowVecSet = []
tfSet = []
testTfSet = []

def get_bow_vec(i, string):
	# Given an index i of the document (review), this will compute the bow vec for it
	if string == 'train':
		s = corpusBow[i]
	else:
		s = testBow[i]
	bowVec = np.zeros(vocabLen)
	normBow = np.zeros(vocabLen)
	pattern = re.compile(r"[0-9]+:[0-9]+")
	match = pattern.findall(s)
	for y in match:
		f = ''
		for j, z in enumerate(y):
			if z != ':':
				f += z
			else:
				idx = j
				break
		frequency = int(y[idx+1:])
		if (int(f)< vocabLen):
			bowVec[int(f)] = 1
			normBow[int(f)] = frequency
		else:
			bowVec[int(f)-1] = 1
			normBow[int(f)-1] = frequency
	t = normBow
	normBow = normBow/(np.sum(normBow))
	return [bowVec, normBow, t]
	pass

"""
# Construct the BOW n-BOW and TF matrices (lists) for training data
print("Constructing the BOW n-BOW and TF matrices (lists) for training data-")
for k in tqdm.trange(12500, 12500 + batchSize):
	d = get_bow_vec(k, 'train')
	bowVecSet.append(d[0])
	nBowVecSet.append(d[1])
	tfSet.append(d[2])

np.save(os.path.join('feature_vecs', 'train_neg_bbow'), bowVecSet)
np.save(os.path.join('feature_vecs','train_neg_nbow'), nBowVecSet)
np.save(os.path.join('feature_vecs','train_neg_tf'), tfSet)

"""
# Construct the BOW n-BOW and TF matrices (lists) for test data
print("Constructing the BOW n-BOW and TF matrices (lists) for test data-")
for k in tqdm.trange(12500, 12500 + batchSize):
	d = get_bow_vec(k, 'test')
	testBowVecSet.append(d[0])
	testnBowVecSet.append(d[1])
	testTfSet.append(d[2])

np.save(os.path.join('feature_vecs', 'test_neg_bbow'), testBowVecSet)
np.save(os.path.join('feature_vecs','test_neg_nbow'), testnBowVecSet)
np.save(os.path.join('feature_vecs','test_neg_tf'), testTfSet)

# Computes the IDF and TFIDF 
docCount = np.zeros(vocabLen)

for column in tqdm.trange(vocabLen):
    docCount[column] += sum(row[column] for row in bowVecSet)

idfVec = np.zeros(vocabLen)
for k in tqdm.trange(vocabLen):
    idfVec[k] = np.log(corpusLen/(1+docCount[k]))
print("The idfVector for the training data:", idfVec)

"""
tfidfSet = []
print("Computing TFIDF for training data:")
for doc in tqdm.trange(batchSize):
    tfidfSet.append(np.multiply(tfSet[doc], idfVec))

np.save(os.path.join('feature_vecs','train_neg_tfidf'), tfidfSet)

"""
testTfidfSet = []
print("Computing TFIDF for test data:")
for doc in tqdm.trange(batchSize):
    testTfidfSet.append(np.multiply(testTfSet[doc], idfVec))

# np.save('train_tfidf0_4000', tfidfSet)
np.save(os.path.join('feature_vecs','test_neg_tfidf'), testTfidfSet)
