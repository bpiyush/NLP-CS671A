import re
import nltk

f = open("./data1.txt", "r+")
raw = f.read();

#Case 1: 'Capital letter -----'
pattern1 = re.compile(r"[^a-z]\s'[A-Z](\s|\S)*?[\.?!,]'")
match1 = pattern1.findall(raw)

print(match1)