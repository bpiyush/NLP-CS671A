import re
import nltk

f = open("./data1.txt")
raw = f.read();
count = 0

'''
comma1 = re.findall(r"['\"](.*,)['\"]", raw)
comma = re.findall(r"['\"](.*?),['\"]", raw)
fullStop = re.findall(r"['\"](.*?).['\"]", raw)
questionMark = re.findall(r"['\"](.*?)\?['\"]", raw)

print(comma1)
print(len(comma1))

print(comma)
print(len(comma))

print(fullStop)
print(len(fullStop))

print(questionMark)
print(len(questionMark))

print(comma[2]==comma1[2])
'''
pattern  = re.compile(r"\s['](\s|\S)*?[']\s")
matches = pattern.finditer(raw)

for match in matches:
	print(match, match.span())
	print(raw[match.span()[0]:match.span()[1]])
	count = count + 1 

print(count)



pat2 = re.compile(r"[.]*['](.*)['][.]*")
mat2 = pattern.finditer(raw)

for mat in mat2:
	print(mat, mat.span())

findWord = re.compile(r"[A-Z][^\\.;]*(said|claimed|cried|replied|answered|stated|snapped|continued|asked|resumed)[^\\.;]*")
matWord = findWord.finditer(raw)
c = 0

matw = re.findall(r"[A-Z][^\\.;]*(said|claimed|cried|replied|answered|stated|snapped|continued|asked|resumed)[^\\.;]*", raw)
print(matw)
for a in matWord:
	print(a)
	print(raw[a.span()[0]:a.span()[1]])
	c += 1
	
print(c) 


sentEnd = re.compile(r"[A-Z][^\\.\?]*(.|!|\?|;)")
sentPat = sentEnd.finditer(raw)
d=0
for b in sentPat:
	print(b)
	print(raw[b.span()[0]:b.span()[1]])
	d +=1


1. r"\s['][A-Z](\s|\S)*?([\.?!][']\s|[\.?!][']\s[a-z].*?[\.])"
2. r"\s'[A-Z](\s|\S)*?[\.?!]'\s"  -------get the quotes

3. r"\s'[A-Z](\s|\S)*?[\.?!]'\s" -----------Type 1
4. r"\s'[A-Z](\s|\S)*?[\.?!]'\s" -----------Type 2





1(a,b) - r"\s[A-Z][\s\(\)a-z;:,'A-Z--]*(.'|.)"
1(c,d) - r"[^a-z]\s[A-Z'][\s\(\)a-z;:,'A-Z--]*(!'|\?')"

r"(\.|\.'|\?|\!)\s['A-Z]"


1a-r"\s'[A-Za-z](\s|\S)*?[\.?!,]'"