import re
import tqdm
from operator import itemgetter


kinds = ['train', 'dev', 'test']
t_str = 'UD_English-EWT/en_ewt-ud-'
files = []

for z in kinds:
	files.append(t_str + z + ".conllu")

def construct_graph(list_of_strs):
	graph = []
	for i,x in enumerate(list_of_strs):
		pa = re.compile(r"\d{1,3}:[a-z]+[\t:]")
		#print(x)
		ma = pa.findall(x)
		y = ma[0]
		y = y[:-1]
		idx = y.index(":")
		rel = y[idx+1:]
		if int(ma[0][:idx]) < i+1:
			k = 'l'
		else:
			k = 'r'
		graph.append([int(ma[0][:idx]), i+1, [k ,rel]])
	return graph
	pass

def construct_configs(graph):
	configs = [[], []]
	vertex_pairs = []
	for x in graph:
		vertex_pairs.append([x[0], x[1]])
	#print(vertex_pairs)
	#Initialize stack, buffer, graph, configs
	new_graph = []
	stack = [0]
	buff = list(range(1,len(graph)+1))

	shift = ['s', '_']

	while len(buff) > 0:
		b0 = buff[0]
		if len(stack) > 0:
			s0 = stack[-1]
			#print(s0, b0)
			if [b0, s0] in vertex_pairs:
				rel = graph[vertex_pairs.index([b0, s0])][2]
				print("Training example:", [stack, buff, new_graph], rel)
				configs[0].extend([stack, buff, new_graph])
				configs[1] += (rel)
				#print(configs[0])
				new_graph.append([b0, s0, rel])
				#print("Removing: ", stack[-1])
				del stack[-1]
				continue

			elif [s0, b0] in vertex_pairs:
				flag = True
				max_idx = len(graph)

				for w in range(max_idx + 1):
					if w != b0:
						if [b0, w] in vertex_pairs:
							rel = graph[vertex_pairs.index([b0, w])][2]
							if [b0, w, rel] not in new_graph:
								flag = False

				if flag:
					rel = graph[vertex_pairs.index([s0, b0])][2]
					print("Training example:",[stack, buff, new_graph], rel)
					configs[0].extend([stack, buff, new_graph])
					configs[1]+=rel
					#print(configs[0])
					new_graph.append([s0, b0, rel])
					#print("Removing: ", buff[0])
					del buff[0]
					continue
				else:
					rel = graph[vertex_pairs.index([s0, b0])][2]
					new_graph.append([s0, b0, rel])
							
		configs[0].extend([stack, buff, new_graph])
		configs[1] += shift
		print("Training example:",[stack, buff, new_graph], shift)
		#print(configs[0])
		stack.append(buff[0])
		del buff[0]
	
	l = sorted(new_graph, key=itemgetter(1))
	#print(l)
	pass


	
def process_file(file_name):
	h = open(file_name)
	raw_file = h.read()

	# Construct DepRel list
	DR_list = []
	pa = re.compile(r"\d{1,3}:[a-z]+[\t:]")
	ma = pa.findall(raw_file)

	for x in ma:
		x = x[:-1]
		idx = x.index(":")
		rel = x[idx+1:]
		DR_list.append(rel)

	DR_list = set(DR_list)
	DR_list = list(DR_list)
	# print(DR_list, len(DR_list))

	doc_id_list = []
	doc_p = re.compile(r"# newdoc id = .+")
	doc_m = doc_p.findall(raw_file)
	for x in doc_m:
		doc_id_list.append(x)
	nraw_file = re.sub(r"# newdoc id = .+", "", raw_file)

	split_data = []
	sent = []
	for x in nraw_file.splitlines():
		if x == '':
			if sent != []:
				split_data.append(sent)
			sent = []
		else:
			sent.append(x)

	# split_data[i] contains all the data of sentence i in form of list of strings
	# print('\n'.join(split_data[3]))
	# g = construct_graph(split_data[3][2:])
	# construct_configs(g)
	for j in tqdm.trange(len(split_data)):
		print('\n'.join(split_data[j]))
		g = construct_graph(split_data[j][2:])
		construct_configs(g)
		print('\n')
	

#process_file(files[0])
