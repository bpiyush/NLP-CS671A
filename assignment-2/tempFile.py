from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'glove.6B/glove.6B.300d.txt'
word2vec_output_file = 'glove.6B.300d.txt.word2vec'

glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.6B.300d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False, limit = 500000)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)