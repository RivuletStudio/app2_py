import numpy as np
import numpy.linalg
import math

class trail_vertex():
	def __init__(self,index,prev_ind,phi):
		self.w = index[0]
		self.h = index[1]
		self.d = index[2]
		self.prev_w = prev_ind[0]
		self.prev_h = prev_ind[1]
		self.prev_d = prev_ind[2]
		self.phi = phi

def euc_distance(x,y):
		distance = math.sqrt((x.w-y.w) ** 2 + (x.h-y.h) ** 2 + (x.d-y.d) ** 2)
		return distance

def edge_distance(max_intensity, x, y):
	# this 10 can be set up as a parameter
	edge_distance = math.fabs(euc_distance(x,y)) * (
		(math.exp(10 * ((1-x.intensity/max_intensity) ** 2))) + (math.exp(10 * ((1-y.intensity/max_intensity) ** 2))) / 2)
	return edge_distance

# insert a a vertex into trial set
def insert_trial_set(trial_set, vertex):
	size = trial_set.size

	# print(size)
	if trial_set.size == 0:
		trial_set = np.append(trial_set,vertex)

	elif vertex.phi >= trial_set[size-1].phi:
		trial_set = np.insert(trial_set, size-1, vertex)

	else:
		index = 0
		for i in trial_set:
			# print(i.w, i.h, i.d,i.dt,i.state)
			if vertex.phi <= i.phi:
				trial_set = np.insert(trial_set,index,vertex)
				break
			index+=1

	return trial_set

def extract_min_from_trial(trial_set):
	print('before extract: ',trial_set.size)
	min_vertex = trial_set[0]
	trial_set = np.delete(trial_set,0)
	
	print('after extract: ',trial_set.size)
	return trial_set,min_vertex


def adjust_in_trial(trial_set,neighbour):
	index = 0

	trial_set = np.delete(trial_set,0)

	index = 0
	for i in trial_set:
		if neighbour.phi <= i.phi:
			trial_set = np.insert(trial_set,index,neighbour)
			break
		index+=1

	return trial_set

#  find average and max intensity
def find_max_intensity(img,size):
	count = 0
	max_intensity = 0
	total_intensity = 0
	not_zero = 0
	max_w = 0
	max_h = 0
	max_d = 0
	for w in range (size[0]):
		for h in range (size[1]):
			for d in range (size[2]):
				if img[w][h][d] > max_intensity:
					max_w = w
					max_h = h
					max_d = d
					max_intensity = img[w][h][d]
				if img[w][h][d] > 0:
					total_intensity+=img[w][h][d]
					not_zero+=1
				count+=1
				# print(img[w][h][d])
	print('total intensity: ',total_intensity)
	print('max intensity: ',max_intensity,max_w,max_d,max_h)
	print('total vertices: ', count)
	return max_intensity

def fastmarching(img,bimg,dt_result,size):
	vertices = np.asarray(img)
	phi = np.array([size[0],size[1],size[2]])
	phi.fill(math.inf)
	state = np.chararray((size[0],size[1],size[2]))
	state[:] = ('FAR')

	parent = np.array([size[0],size[1],size[2]])
	for w in range (size[0]):
		for h in range (size[1]):
			for d in range (size[2]):
				parent.itemset((w,h,d),w)

	location = find_max_intensity(img,size)
	print(location[1],location[2],location[3])
	state[location[1]][location[2]][location[3]] = 'ALIVE'
	phi[location[1]][location[2]][location[3]] = 0.0

	trail_set = []
	root = trail_vertex([location[1],location[2],location[3]],[location[1],location[2],location[3]],0.0)
	trail_set.append(root)

