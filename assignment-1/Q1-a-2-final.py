import re

f = open("./dataset.txt")
dataset = f.read()

g = open("./FullDataset.txt")
FullDataset = g.read()

h = open("./testData.txt")
testData = h.read()

toBeTagged = testData #File to be tagged
toBeTagged_list = list(toBeTagged)

pattern = re.compile(r"([^A-Z]|I)[^A-Z](\.|\.'|\?|\?'|!|!'|-')\s+[A-Z'*][^\.]")
match = pattern.finditer(toBeTagged)

no_of_tags = 0
tag_indices = []
tags_domain_list = []

for x in match:
	no_of_tags += 1
	a = x.span()[0]
	b = x.span()[1]
	tag_indices.append((a,b))
	tags_domain_list.append(toBeTagged[a:b])

new_tagged_list = []
for x in tags_domain_list:
	t = list(x)
	if t[3] == "'":
		t.insert(4, "</s>")
		i = 5
	else:
		t.insert(3, "</s>")
		i = 4
	#print(t)
	if t[i] == " ":
		t.insert(i, "<s>")
	else:
		j = i 
		while t[j] in ["\n", "*"]:
			j += 1 
		t.insert(j, "<s>")
	#print(t)
	new_tagged_list.append("".join(t))

#print(new_tagged_list)

finalOut = []

for idx in range(no_of_tags):
	if idx == 0:
		finalOut.append(toBeTagged[:tag_indices[idx][0]])
	else:
		finalOut.append(toBeTagged[tag_indices[idx-1][1]:tag_indices[idx][0]])

finalOut.append(toBeTagged[tag_indices[no_of_tags-1][1]:])



for i in range(no_of_tags):
	finalOut[i] += new_tagged_list[i]

result = "".join(finalOut)
print(result)