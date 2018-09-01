import re
import nltk

f = open("./expt-data.txt", "r+")
raw = f.read();
count = 0

p = re.compile(r"(\.|\.'|\?|\!)\s[A-Z']")
m = p.finditer(raw)
m1 = p.findall(raw)

saw = raw
for i in reversed(m1):
	print(i)
	b = i.span()
	# f.seek(b[0])
	# f.write('</s>')
	saw = saw[:b[0]] + ' </s> ' + saw[b[0]:]

print(saw)