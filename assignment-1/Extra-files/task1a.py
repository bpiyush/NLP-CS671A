import re
import nltk

f = open("./data1.txt")
raw = f.read();
count = 0

# Extract all sentences
sentP = re.compile(r"[A-Z][^\.?!;]+[\.?!;]")
sentM = sentP.findall(raw)

# print(sentM, len(sentM))
x = '-------'
for a in sentM:
	print(x,a)

print(len(sentM))

g = open("./data2.txt")
testraw = g.read();

y = '+++++++++'
sentP1 = re.compile(r"[A-Z'][^\.?!]*[\.'?!]")
sentM1 = sentP1.finditer(testraw)

for b in sentM1:
	print(y,testraw[b.span()[0]:b.span()[1]])

sm = re.compile(r"[A-Z'][a-zA-Z]+('\w+')[\s\S]*[\.?!]")