import re
import numpy as np

f = open("./dataset.txt", "r+")
raw = f.read()

h = open("./FullDataset.txt", "r+")
nraw = h.read()

m = open("./testData.txt")
testData = m.read()

toBeTagged = testData # name of the file containing the data to be tagged

pattern = re.compile(r"[^A-Z][^A-Z](\.|\.'|\?|\?'|!|!')\s+[A-Z'][^\.]")
match = pattern.finditer(toBeTagged)

y = []
s = list()

for x in match:
	a = x.span()[0]
	b = x.span()[1]
	y.append((a,b))
	s.append(toBeTagged[a:b])

newStr = []

for t in s:
	tempStr = list(t)
	if tempStr[2] in ('.', '?', '!') and tempStr[3] not in ("'"):
		if tempStr[3]==' ':
			tempStr.insert(3, '</s> <s>')
		if tempStr[3] in ('\n') and tempStr[4] not in ('\n'):
			tempStr.insert(3, '</s>')
			tempStr.insert(5, '<s>')
		if tempStr[3] in ('\n') and tempStr[4] in ('\n'):
			tempStr.insert(3, '</s>')
			tempStr.insert(6, '<s>')
	if tempStr[2] in ('.', '?', '!') and tempStr[3] in ("'"):
		if tempStr[4]==' ':
			tempStr.insert(4, '</s> <s>')
		if tempStr[4] in ('\n') and tempStr[5] not in ('\n'):
			tempStr.insert(4, '</s>')
			tempStr.insert(6, '<s>')
		if tempStr[4] in ('\n') and tempStr[5] in ('\n'):
			tempStr.insert(4, '</s>')
			tempStr.insert(7, '<s>')
	q = "".join(tempStr)
	newStr.append(q)

finalStr = []

for i in range(len(y)):
	if i==0:
		finalStr.append(toBeTagged[0:y[i][0]])
	else:
		if i==len(y)-1:
			finalStr.append(toBeTagged[y[i][1]:len(toBeTagged)-1])
		else:
			finalStr.append(toBeTagged[y[i-1][1]:y[i][0]])

finalStr1 = []
for j in range(len(y)):
	finalStr1.append(finalStr[j]+newStr[j])

finalStr1.insert(len(y)-1, toBeTagged[y[len(y)-2][1]:y[len(y)-1][0]])
f = len(finalStr1)
finalStr1[f-2] += (newStr[len(y)-1])
finalStr1[f-1] = toBeTagged[y[len(y)-1][1]:]


result = "".join(finalStr1)

print(result)

