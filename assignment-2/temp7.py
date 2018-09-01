import re
import numpy as np
import gensim.models as gsm
import os
import stop_words
import gensim
import tqdm
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument

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

sentences = []
sentence_id = []
for z in data:
    y = split_doc_into_sent(z)
    if len(sentence_id) >= 1:
        h = sentence_id[len(sentence_id)-1]
    else:
        h = 0
    sentence_id.append(h + len(y)) # Number of sentences in given document
    for t in y:
        sentences.append(t)


print(sentence_id)
print("Number of Sentences found:", len(sentences))
print(len(sentence_id))
for i in range(9):
    print(sentences[i])

