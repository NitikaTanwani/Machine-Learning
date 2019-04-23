import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return


# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	#print(train_data[0].tags)

	###################################################
	# Edit here
	#pi=np.empty(shape=[0, len(train_data)])
	pi=[]
	dic_tags={}
	total_occu=len(train_data)
	#print(total_occu)

	for i in range(0,total_occu):
		if train_data[i].tags[0] in dic_tags:
			dic_tags[train_data[i].tags[0]]+=1
		else:
			dic_tags[train_data[i].tags[0]]=1
	#print(dic_tags)
	itr=0
	
	for j in range(0,len(tags)):
		
		if tags[j] in dic_tags.keys():
			pi.append(dic_tags[tags[j]]/float(total_occu))
			#print(tags[j])
			#print(dic_tags[tags[j]]/float(total_occu))
		else:
			pi.append(0)
			#print(tags[j])
	
	#generating transition matrices A
	transition_matrix={}
	sentec=[]
	for i in range(0,len(train_data)):
		sentec=train_data[i].tags
		#print(train_data[i].tags)
		for i in range(0,len(sentec)-1):
			if sentec[i] in transition_matrix.keys():
				transition_matrix[sentec[i]].append(sentec[i+1])
			else:
				transition_matrix[sentec[i]]=[sentec[i+1]]
				#transition_matrix[sentec[i]].add(sentec[i+1])
	#print(transition_matrix)

	new_dict={}
	for key,val in transition_matrix.items():
		new_dict=construct_dict(val)
		transition_matrix[key]=new_dict
	#print(transition_matrix)

	#calculating probabilities
	for key,value in transition_matrix.items():
		val=calc_prob(transition_matrix[key])
		transition_matrix[key]=val
	#print(transition_matrix)

	#emission probability calculataing Bs
	emission_matrix={}
	for i in range(0,len(train_data)):
		lis_words=train_data[i].words
		lis_tags=train_data[i].tags
		for i in range(0,len(lis_words)):
			if lis_tags[i] not in emission_matrix.keys():
				emission_matrix[lis_tags[i]]=[lis_words[i]]
			else:
				emission_matrix[lis_tags[i]].append(lis_words[i])

	#print(emission_matrix)

	for key,value in emission_matrix.items():
		new_dict=construct_dict(emission_matrix[key])
		emission_matrix[key]=new_dict

	#print(emission_matrix)

	for key,value in emission_matrix.items():
		val=calc_prob(emission_matrix[key])
		emission_matrix[key]=val
	#print(emission_matrix)


	#converting dictionary to numpy arrays for transition
	#pi=np.asarray(pi)
	#print(transition_matrix)
	#emiss_pass=emission_matrix
	emiss_pass={}
	for key,value in emission_matrix.items():
		emiss_pass[key]=value

	state_dic={}
	count=0
	for i in range(0,len(tags)):
		state_dic[tags[i]]=count
		count+=1

	#print(state_dic)
	obj_dic=get_object_dic(emiss_pass)
	#state_dic=get_state_dic(emission_matrix)
	#print(state_dic)

	# if you print transiion and meission matrix here, you can see them before consverting 
	# it into numpy array
	#print(emission_matrix)
	#print(transition_matrix)
	#print(transition_matrix["NOUN"]["."])
	#print(emission_matrix["NOUN"]["concept"])

	trans_array=np.zeros([len(tags),len(tags)])

	#print(transition_matrix["X"])

	for k1 in tags:
		if k1 in transition_matrix:
			for k2 in tags:
				if k2 in transition_matrix[k1]:
					trans_array[state_dic[k1]][state_dic[k2]]=transition_matrix[k1][k2]



	#print(emission_matrix["ADV"]["all"])
	emis_array=np.zeros([len(tags),len(obj_dic)])
	for k1 in tags:
		if k1 in emission_matrix:
			for k2 in obj_dic:
				if k2 in emission_matrix[k1]:
					emis_array[state_dic[k1]][obj_dic[k2]]=emission_matrix[k1][k2]

	#print(emis_array[state_dic["ADV"]][obj_dic["all"]])
	#transition_matrix=convert_numpy(transition_matrix,state_dic)
	#emission_matrix=convert_numpy(emission_matrix,state_dic)
	#print(state_dic)
	#obj_dic={}
	#print(trans_array[state_dic["NOUN"]] [state_dic["."]])
	#print(state_dic["NOUN"])
	#print(state_dic["."])
	#print(trans_array[6] [0])
	#print(emission_matrix[state_dic["NOUN"]][obj_dic["concept"]])

	# After converting it into required format
	#print(emission_matrix)
	#print(transition_matrix["NOUN"]["NOUN"])
	
	'''model.pi=np.array(pi)
	model.A=np.asarray(trans_array)
	model.B=np.asarray(emis_array)
	model.obs_dict=obj_dic
	model.state_dict=state_dic'''

	model=HMM(np.array(pi),np.array(trans_array),np.array(emis_array),obj_dic,state_dic)
	#return model

	
	#print(transition_matrix)
	#print(pi)
	#print(obj_dic)
	#print(state_dic)

	#final result
	
	#print(transition_matrix)
	#print(emission_matrix)





	###################################################
	return model

def get_state_dic(emission_matrix):
	res={}
	count=0
	for key,value in emission_matrix.items():
		res[key]=count
		count+=1
	return res

def convert_numpy(dic,state_dic):
	#index={}
	count=0
	new_d={}
	final_tran=[]
	for key,value in dic.items():
		state_dic[key]=count
		count+=1
		new_d=value
		inter=[]
		for key,value in new_d.items():
			inter.append(value)
		final_tran.append(inter)
	return (np.asarray(final_tran))
	#print(index)

def get_object_dic(emis):
	obj_dic={}
	inner_d={}
	for key,value in emis.items():
		inner_d=value
		count=0
		for k1,v1 in inner_d.items():
			obj_dic[k1]=count
			count+=1
	return obj_dic





def construct_dict(val):
	new_dict={}
	for i in range(0,len(val)):
		if val[i] in new_dict.keys():
			new_dict[val[i]]+=1
		else:
			new_dict[val[i]]=1
	return new_dict

def calc_prob(value):
	total=0
	for key,val in value.items():
		total+=val
	if(total==0):
		return value

	for key,val in value.items():
		value[key]=val/float(total)
	#print(value)
	return value


# TODO:
def speech_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	#print(model)
    
	###################################################
	return tagging
def create_empty(b):
	return np.empty((b.shape[0],1))

def send_concate(b,extra):
	return np.concatenate((b,extra),axis=1)
