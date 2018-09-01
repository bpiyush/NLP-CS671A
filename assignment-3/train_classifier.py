import re
import tqdm
from operator import itemgetter
import get_data
import ast
import gensim.models as gsm
import numpy as np
from numpy import array
from nltk import word_tokenize
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.txt.word2vec'


glove2word2vec(glove_input_file, word2vec_output_file)

print("Loading model...- ")
# Load the Google word2vec model
w2v = gsm.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
# load the Stanford GloVe model
gloveModel = gsm.KeyedVectors.load_word2vec_format('glove.6B.50d.txt.word2vec', binary=False, limit = 500000)
print("Models Loaded.")

kinds = ['train', 'dev', 'test']
t_str = 'UD_English-EWT/en_ewt-ud-'
files = []

for z in kinds:
	files.append(t_str + z + ".conllu")

f = open('training_data.txt')
raw = f.read()

train_file = open(files[0]).read()


def get_graph_len(x):
	count = 0
	for y in x:
		if y[0] != 'T':
			count += 1
		else:
			break
	return count - 2
	pass

# Construct DepRel list
DR_list = []
pa = re.compile(r"\d{1,3}:[a-z]+[\t:]")
ma = pa.findall(raw)

for x in ma:
	x = x[:-1]
	idx = x.index(":")
	rel = x[idx+1:]
	DR_list.append(rel)

DR_list = set(DR_list)
DR_list = list(DR_list)

set_of_classes = []
set_of_classes.append(['s', '_']) # Adding shift opertion on classes

for x in DR_list:
	set_of_classes.append(['l', x])
	set_of_classes.append(['r', x])

#print(set_of_classes, len(set_of_classes))


train_ex = []

h = raw.splitlines()

split_data = []
sent = []
for i,x in enumerate(h):
	if i < len(h) - 1:
		if x == '' and h[i+1] == '':
			if sent != []:
				split_data.append(sent)
			sent = []
		else:
			sent.append(x)


for i in range(1,len(split_data)):
	del split_data[i][0]
# print(split_data[2])

# Construct POS tag set
POS_tags = []
pat = re.compile(r"[^A-Z\dI]\t[A-Z]+\t")
mat = pat.findall(train_file)

for x in mat:
	POS_tags.append(x[2:-1])

POS_tags = set(POS_tags)
POS_tags = list(POS_tags)
del POS_tags[POS_tags.index('X')]
del POS_tags[POS_tags.index('I')]
#print(POS_tags, len(POS_tags))

def get_POS_idx(string):
	patt = re.compile(r"[^A-Z\dI]\t[A-Z]+\t")
	matc = pat.findall(string)
	if matc != []:
		if len(matc[0]) > 2:
			if matc[0][2:-1] in ['I', 'X']:
				g = -100
			else:
				g = POS_tags.index(matc[0][2:-1])
		else:
			g = -100
	else:
		g = -100
	return g
	pass

def get_deps(graph, s):
	for c in graph:
		if s == c[1]:
			return max(set_of_classes.index(c[2]),-100)
	pass
def get_word(string):
	s= word_tokenize(string)
	h = np.zeros(300)
	if s[2] in w2v.vocab:
		h += w2v[s[2]]
	return h
	pass
def get_word_glove(string):
	s= word_tokenize(string)
	h = np.zeros(50)
	if s[2] in gloveModel.vocab:
		h += gloveModel[s[2]]
	return h
	pass
def get_feature_vec(details, config):
	keys = [0,1,2] #Important indices to work on stack and buffer
	stack = config[0]
	buff = config[1]
	curr_graph = config[2]
	vec = [] # Conatins all the elements in the feature vector
	# Feature vectors structure: (POS(s_0), POS(s_1), POS(s_2),POS(b_0), POS(b_1), POS(b_2), DEP(s_0), DEP(s_1), DEP(s_0),
	# DEP(s_1), DEP(s_2), DEP(b_0), DEP(b_1), DEP(b_2), word_vector(s_0) and so so)
	#print(len(details), stack, buff)
	# Adding POS values
	for i in keys:
		if i in range(len(stack)):
			vec.append(get_POS_idx(details[stack[i]]))
		else:
			vec.append(-100)
		if i in range(len(buff)):
			vec.append(get_POS_idx(details[buff[i]-1]))
		else:
			vec.append(-100)

	# Adding DEP rel values
	for i in keys:
		if i in range(len(stack)):
			vec.append(get_deps(curr_graph, stack[i]))
		else:
			vec.append(-100)
		if i in range(len(buff)):
			vec.append(get_deps(curr_graph, buff[i]))
		else:
			vec.append(-100)

	# Adding word embeddings
	v = np.zeros(300)
	for i in keys:
		if i in range(len(stack)):
			v += get_word(details[stack[i]])
			#v[a:c] += get_word_glove(details[stack[i]])
		if i in range(len(buff)):
			#v[c:b] = get_word_glove(details[buff[i]])
			v += get_word(details[buff[i]-1])
	for i in range(300):
		vec.append(v[i])
	#print(array(vec))
	return array(vec)
	pass

training_X = []
training_y = []
for p in tqdm.trange(len(split_data[:100])):
	#sent_configs = []
	x = split_data[p]
	l = get_graph_len(x) 
	# print(l)
	gx = get_data.construct_graph(list(x[2:l]))
	#print(gx)
	
	for y in x[l+2:]:
		# print(y)
		strs = y[18:].replace('[','').split('],')
		stack = ast.literal_eval('[' + strs[0] + ']')
		buff = ast.literal_eval('[' + strs[1] + ']')
		# print("Stack:" , stack)
		# print("Buffer:" , buff)

		p1 = re.compile(r"\[\d{1,2}, \d{1,2}, \['r', '[a-z]+'\]\]")
		m1 = p1.findall(y)
		p2 = re.compile(r"\[\d{1,2}, \d{1,2}, \['l', '[a-z]+'\]\]")
		m2 = p2.findall(y)
		curr_graph = ast.literal_eval(str(m1+m2))
		for i in range(len(curr_graph)):
			curr_graph[i] = ast.literal_eval(curr_graph[i])
		# print("Graph:" , curr_graph)
		config = [stack, buff, curr_graph]
		#sent_configs.append(config)
		train_ex.append([stack, buff, curr_graph])

		p3 = re.compile(r"[^,] \['r', '[a-z]+'\]")
		m3 = p3.findall(y)
		p4 = re.compile(r"[^,] \['l', '[a-z]+'\]")
		m4 = p4.findall(y)
		p5 = re.compile(r"[^,] \['s', '_'\]")
		m5 = p5.findall(y)
		z = m3 + m4 + m5
		train_label = ast.literal_eval(str(z[0][2:]))
		training_y.append(set_of_classes.index(train_label))
		# print("Training label:" , train_label)

		f = get_feature_vec(x[2:l+2], config)
		training_X.append(f)
		#print(f)


print("Size of training data: ", len(training_X), len(training_y))
for i,x in enumerate(training_X):
	for j, y in enumerate(x):
		if y == None:
			x[j] = -100


kf = KFold(n_splits=5)
training_X = np.array(training_X)
training_y = np.array(training_y)
for train_index, test_index in kf.split(training_X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = training_X[train_index], training_X[test_index]
    y_train, y_test = training_y[train_index], training_y[test_index]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 6, 2), random_state=1)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print("Accuracy obtained with Neural Nets: ",accuracy_score(y_test, predicted))



# I am working with only thousand examples
"""
rand_smpl = [ split_data[i] for i in sorted(random.sample(xrange(len(split_data[:1000])), 50)) ]
train_smpl = [item for item in split_data if item not in rand_smpl]

print(rand_smpl[0], train_smpl[0])

# Training the oracle
print("Beginning training:")
training_X = np.array(training_X)
training_y = np.array(training_y)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 4), random_state=1)
clf.fit(training_X, training_y)

def shift(config):
	[stack, buff, graph] = config
	stack.append(buff[0])
	del buff[0]
	return [stack, buff, graph]
	pass

def arc_left(config, opn):
	[stack, buff, graph] = config
	graph.append([buff[0], stack[-1], opn])
	del stack[-1]
	return [stack, buff, graph]
	pass

def arc_right(config, opn):
	[stack, buff, graph] = config
	graph.append([stack[-1], buff[0], opn])
	buff = [stack[-1]] + buff
	del stack[-1]
	return [stack, buff, graph]
	pass

# Parser function
def parser(sent_idx):
	x = split_data[sent_idx]
	l = get_graph_len(x) 
	# print(l)
	# original parse
	gx = get_data.construct_graph(list(x[2:l]))
	stack = [0]
	buff = list(range(1,l + 1))
	graph = []
	config = [stack, buff, graph]
	while len(buff) >= 1:
		f = get_feature_vec(x[2:l+2], config)
		for i,k in enumerate(f):
			if k == None:
				f[i] = -100

		opn_idx = clf.predict([f])
		print(opn_idx[0])
		opn = set_of_classes[opn_idx[0]]

		if opn[0] == 'l':
			config = arc_left(config, opn)
		elif opn[0] == 'r':
			config = arc_right(config, opn)
		else:
			config = shift(config)
		print(config)
	return config
	pass

print(parser(0))
"""

"""
import json
with open('feat_vecs.txt', 'w') as outfile:
    json.dump([training_X, training_y], outfile)
sentences = [h[i:j] for i,j in zip(indices, indices[1:]+[None])]
print(sentences[0])
"""