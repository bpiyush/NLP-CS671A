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

print("Loading model of sentence2vec:")
model = Doc2Vec.load('./imdb_sent_vec.s2v')

print(model['train_posdoc_id0sent_id1'])