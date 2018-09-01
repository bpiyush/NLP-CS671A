import re

f = open("./dataset.txt")
dataset = f.read();
dataset_list = list(dataset)

g = open("./FullDataset.txt")
FullDataset = g.read();
FullDataset_list = list(FullDataset)

#file that needs to be edited
toBeConverted = dataset
toBeConverted_list = dataset_list

#Case 1: Type: 'Capital letter -----'
pattern1 = re.compile(r"[^a-z']\s'[A-Z](\s|\S)*?[\.?!, -]'")
match1 = pattern1.finditer(toBeConverted)

for x in match1:
	a = x.span()[0]
	b = x.span()[1]
	toBeConverted_list[a+2] = '"'
	toBeConverted_list[b-1] = '"'

res = "".join(toBeConverted_list)
res_list = list(res)

#Case 2: Type:, 'small letter -----'
pattern2 = re.compile(r"[,:;]\s'[a-z](\s|\S)*?[\.?!,-]'\s[^a-z]")
match2 = pattern2.finditer(res)

for x in match2:
	a = x.span()[0]
	b = x.span()[1]
	res_list[a+2] = '"'
	res_list[b-3] = '"'

result = "".join(res_list)
print(result)