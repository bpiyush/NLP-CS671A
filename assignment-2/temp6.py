import os
import re
import numpy as np
import stop_words
import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import tqdm

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
    # file_list has the names of all the files in the folder_name
    file_list = [folder_name+'/'+name for name in os.listdir(folder_name) if name.endswith('txt')]
    print(file_list)
    file_list = sort_custom(file_list)
    print(file_list)
    pass

get_doc_list(folders[0])