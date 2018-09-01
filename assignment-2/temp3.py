import re

doc = "Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as 'Teachers'. My 35 years in the teaching profession lead me to believe that Bromwell High's satire is much closer to reality than is 'Teachers'. The scramble to survive financially, the insightful students who can see right through their pathetic teachers' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High. A classic line: INSPECTOR: I'm here to sack one of your teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isn't!"

def split_doc_into_sent(doc):
    pattern = re.compile(r"([^A-Z]|I)[^A-Z\.](\.|\.'|\?|\?'|!|!'|-')\s+[A-Z'*][^\.]")
    match = pattern.finditer(doc)
    split_idx = []
    split_idx.append(0)
    for x in match:
        split_idx.append(x.span()[0]+3)
    parts = [doc[i:j] for i,j in zip(split_idx, split_idx[1:]+[None])]
    return parts
    pass

y = split_doc_into_sent(doc)
print(y)