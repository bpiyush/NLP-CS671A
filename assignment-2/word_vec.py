import gensim.models as gsm
import numpy as np
import tqdm
import os

from gensim.scripts.glove2word2vec import glove2word2vec

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

batchSize = 500

folder = 'feature_vecs/'

def form_wv(datatype, label):
	bbow  = np.load(folder + datatype + '_' + label + '_' + 'bbow.npy')
	tfidf = np.load(folder + datatype + '_' + label + '_' + 'tfidf.npy')
	wordVecAvg = []
	weighted_wva = []
	gloveAvg = []
	weightedGloveAvg = []
	# quantity = (bool(label == "neg"))*12500
	print("Constructing word vectors for "+ datatype + "ing with label "+ label)
	for doc in tqdm.trange(batchSize):
		docAvg = np.zeros(300)
		weightedDocAvg = np.zeros(300)
		gloveDocAvg = np.zeros(300)
		weightedGloveDocAvg = np.zeros(300)
		factor = np.sum(bbow[doc])
		for j,x in enumerate(bbow[doc]):
			if x == 1:
				if corpusVocab[j] in w2v.vocab:
					wv = w2v[corpusVocab[j]]	
					docAvg += wv
					weightedDocAvg += (tfidf[doc][j])*wv
				if corpusVocab[j] in gloveModel.vocab:
					gVec = gloveModel[corpusVocab[j]]
					gloveDocAvg += gVec
					weightedGloveDocAvg += (tfidf[doc][j])*gVec
		wordVecAvg.append(docAvg/factor)
		weighted_wva.append(weightedDocAvg/factor)
		gloveAvg.append(gloveDocAvg/factor)
		weightedGloveAvg.append(weightedGloveDocAvg/factor)
	return [wordVecAvg, weighted_wva, gloveAvg, weightedGloveAvg]
	pass

train_pos_vec = form_wv('train', 'pos')
np.save(os.path.join('feature_vecs', 'train_pos_vec'), train_pos_vec)
train_neg_vec = form_wv('train', 'neg')
np.save(os.path.join('feature_vecs', 'train_neg_vec'), train_neg_vec)
test_pos_vec = form_wv('test', 'pos')
np.save(os.path.join('feature_vecs', 'test_pos_vec'), test_pos_vec)
test_neg_vec = form_wv('test', 'neg')
np.save(os.path.join('feature_vecs', 'test_neg_vec'), test_neg_vec)


