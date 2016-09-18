import numpy as np
import numpy.linalg
import math


class vertex():
	def __init__(self,_w,_h,_d,_dt,_intensity,_state):
		self.w = _w
		self.h = _h
		self.d = _d
		self.dt= _dt
		self.intensity = _intensity
		self.state = _state
		self.parent = [_w,_h,_d]
		self.phi = math.inf
		self.neighbours = np.empty(6,dtype=vertex)

	def add_neighbours(self,vertex):
		self.neighbours.append(vertex)
		vertex.neighbours.append(self)

def euc_distance(x,y):
		distance = math.sqrt((x.w-y.w) ** 2 + (x.h-y.h) ** 2 + (x.d-y.d) ** 2)
		return distance

def edge_distance(max_intensity, x, y):
	# this 10 can be set up as a parameter
	edge_distance = math.fabs(euc_distance(x,y)) * (
		(math.exp(10 * ((1-x.intensity/max_intensity) ** 2))) + (math.exp(10 * ((1-y.intensity/max_intensity) ** 2))) / 2)
	return edge_distance


# initialize the fast marching tree
def initialize(size,img,dtimg,bimg):
	vertices = np.empty((size[0],size[1],size[2]),dtype=vertex)

	count = 0
	for w in range (size[0]):
		for h in range (size[1]):
			for d in range (size[2]):
				# ignore all background?
				state = 'FAR'
				if dtimg[w][h][d] == 0:
					state = 'STOP'
					
				element = vertex(w, h, d, dtimg[w][h][d], img[w][h][d], state)
				vertices[w][h][d] = element
				# set parent to None?
				vertices[w][h][d].parent = None

	return vertices

#  find average and max intensity
def find_intensity(img,size):
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
	average_intensity = total_intensity/(not_zero)
	return average_intensity,max_intensity

# find centroid of soma
# def find_cource(bimg):


# find trial set
def find_trial_set(vertices,max_w,max_h,max_d,size):
	# set source which has the largest dt value
	vertices[max_w][max_h][max_d].state = 'ALIVE'

	trial_set = np.array([],dtype=vertex)
	count = 0
	
	neighbours = get_neighbours(vertices,max_w,max_h,max_d,size) 
					
	# set neighbours
	# vertices[max_w][max_h][max_d].neighbours = neighbours

	# find trial set
	for i in neighbours:
		i.state = 'TRIAL'
		# set parent
		i.parent = vertices[max_w][max_h][max_d]
		# sort trial set
		trial_set = insert_trial_set(trial_set,i)
		count+=1
		print('Actual' + str(trial_set.size))
		print('Expected' + str(count))

	return trial_set

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


def find_index_in_trial(trial_set,neighbour,w,h,d):
	index = 0
	print(type(neighbour))
	for i in trial_set:
		if neighbour.w == w and neighbour.h == h and neighbour.d == d:
			return index
		index+=1
	return None

# get neighbour vertices of a vertex
def get_neighbours(vertices,w,h,d,size):
	neighbours = np.empty(6,dtype=vertex)
	if (w-1) >= 0:
		if vertices[w-1][h][d].state != 'ALIVE':
			neighbours[0] = vertices[w-1][h][d]
	if (w+1) < size[0]:
		if vertices[w+1][h][d].state != 'ALIVE':
			neighbours[1] = vertices[w+1][h][d]

	if (h-1) >= 0:
		if vertices[w][h-1][d].state != 'ALIVE':
			neighbours[2] = vertices[w][h-1][d]
	if (h+1) < size[1]:
		if vertices[w][h+1][d].state != 'ALIVE':
			neighbours[3] = vertices[w][h+1][d]

	if (d-1) >= 0:
		if vertices[w][h][d-1].state != 'ALIVE':
			neighbours[4] = vertices[w][h][d-1]
	if (d+1) < size[2]:	
		if vertices[w][h][d+1].state != 'ALIVE':
			neighbours[5] = vertices[w][h][d+1]

	return neighbours

def update_distance(vertices, trial_set, max_intensity, size):
	trial_size = 1
	loop_count = 0
	while(loop_count <= 3):
		for x in trial_set:
			print('trial_size1	: ' + str(trial_set.size))	
			print('trial vertex to be extracted: ',x.w,x.h,x.d)
			neighbours = get_neighbours(vertices, x.w, x.h, x.d, size)
			trial_set = extract_min_from_trial(trial_set)
			print(neighbours.size)
			for y in neighbours:
				if y is not None:
					print('neighbours: ',y.w,y.h,y.d,y.state)
					if y.state == 'FAR':
						y.parent = x
						y.state = 'TRIAL'
						print('before insert: ',trial_set.size)
						trial_set = insert_trial_set(trial_set,y)
						print('after insert: ',trial_set.size)
					elif y.state == 'TRIAL':
						if (x.dt + edge_distance(max_intensity,x,y) < x.dt):
							y.parent = x
		trial_size = trial_set.size


def fastmarching_dt_tree(img,bimg,dt_result,size,max_w,max_h,max_d,max_intensity):

	vertices = np.empty((size[0],size[1],size[2]),dtype=vertex)

	for w in range (size[0]):
		for h in range (size[1]):
			for d in range (size[2]):
				state = 'FAR'
				element = vertex(w, h, d, dt_result[w][h][d], img[w][h][d], state)
				vertices[w][h][d] = element

	vertices[max_w][max_h][max_d].state = 'ALIVE'
	vertices[max_w][max_h][max_d].phi = 0.0
	trial_set = np.array([],dtype=vertex)

	trial_set = insert_trial_set(trial_set,vertices[max_w][max_h][max_d])

	while(trial_set.size != 0):
		trial_set,min_vertex = extract_min_from_trial(trial_set)
		
		# long min_ind = min_elem->img_ind;
		# long prev_ind = min_elem->prev_ind;
		# delete min_elem;

		# parent[min_ind] = prev_ind;


		min_vertex.state = 'ALIVE'
		i = min_vertex.w
		j = min_vertex.h
		k = min_vertex.d

		for kk in range (-1,2):
			d = k+kk
			if(d < 0 or d >= size[2]):
				continue
			for jj in range (-1,2):
				h = j+jj
				if(h < 0 or h >= size[1]):
					continue
				for ii in range (-1,2):
					w = i+ii
					if(w < 0 or w >= size[0]):
						continue

					neighbour = vertices[w][h][d]
					if (neighbour.state != 'ALIVE'):
						# not square?
						new_dist = min_vertex.phi + (1.0 - (img[w][h][d]/max_intensity)*1000.0)
						parent_index = [min_vertex.w,min_vertex.h,min_vertex.d]

						if(neighbour.state == 'FAR'):
							neighbour.phi = new_dist
							neighbour.state = 'TRIAL'
							trial_set = insert_trial_set(trial_set,neighbour)
						elif(neighbour.state == 'TRIAL'):
							if (neighbour.phi > new_dist):
								neighbour.phi = new_dist
								index = find_index_in_trial(trial_set,neighbour,w,h,d)
								print(index)

	


		