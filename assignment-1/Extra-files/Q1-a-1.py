import re
import nltk
from shutil import copyfile

f = open("./dataset.txt", "r+")
raw = f.read();

copyfile("./dataset.txt", "./temp-data1.txt")

g = open("./temp-data1.txt", "r+")
midRaw = g.read()

c = 0
d = 0

#Case 1: Type: 'Capital letter -----'
pattern1 = re.compile(r"[^a-z']\s'[A-Z](\s|\S)*?[\.?!,]'")
match1 = pattern1.finditer(raw)

for x in match1:
	c += 1
	g.seek(x.span()[0]+2)
	g.write('"')
	g.seek(x.span()[1]-1)
	g.write('"')

#Case 2: Type:, 'small letter -----'
pattern2 = re.compile(r"[,:;]\s'[a-z](\s|\S)*?[\.?!,]'\s[^a-z]")
match2 = pattern2.finditer(midRaw)

for x in match2:
	d += 1
	g.seek(x.span()[0]+2)
	g.write('"')
	g.seek(x.span()[1]-3)
	g.write('"')
