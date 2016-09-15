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
		self.parent = self
		self.neighbours = np.empty(6,dtype=vertex)

	def add_neighbours(self,vertex):
		self.neighbours.append(vertex)
		vertex.neighbours.append(self)

	def euc_distance(self,vertex):
		distance = math.sqrt((self.w-vertex.w) ** 2 + (self.h-vertex.h) ** 2 + (self.d-vertex.d) ** 2)
		return distance

	def geodesic_distance(max_intensity,vertex):
		# this 10 can be set up as a parameter
		return math.exp(10 * ((1-vertex.intensity/max_intensity) ** 2))

	def edge_distance(self, max_intensity, x, y):
		edge_distance = math.fabs(x.euc_distance(y)) * (
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
				if dtimg[w][h][d] > 0:
					state = 'FAR'
					element = vertex(w, h, d, dtimg[w][h][d], img[w][h][d], state)
					vertices[w][h][d] = element
					# set parent to None?
					vertices[w][h][d].parent = None

	return vertices

#  find average and max intensity
def find_intensity(img,size):
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
				# print(img[w][h][d])
	print('total intensity: ',total_intensity)
	print('max intensity: ',max_intensity,max_w,max_d,max_h)
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

	elif vertex.dt >= trial_set[size-1].dt:
		trial_set = np.insert(trial_set, size-1, vertex)

	else:
		index = 0
		for i in trial_set:
			# print(i.w, i.h, i.d,i.dt,i.state)
			if vertex.dt <= i.dt:
				trial_set = np.insert(trial_set,index,vertex)
				break
			index+=1

	return trial_set

def extract_min_from_trial(trial_set):
	trial_set = np.delete(trial_set,0)
	return trial_set


# get neighbour vetices of a vertex
def get_neighbours(vertices,w,h,d,size):
	neighbours = np.empty(6,dtype=vertex)
	if (w-1) >= 0:
		neighbours[0] = vertices[w-1][h][d]
	if (w+1) < size[0]:
		neighbours[1] = vertices[w+1][h][d]

	if (h-1) >= 0:
		neighbours[2] = vertices[w][h-1][d]
	if (h+1) < size[1]:
		neighbours[3] = vertices[w][h+1][d]

	if (d-1) >= 0:
		neighbours[4] = vertices[w][h][d-1]
	if (d+1) < size[2]:
		neighbours[5] = vertices[w][h][d+1]

	return neighbours

def update_distance(vertices, trial_set, max_intensity, size):
	while(trial_set.size != 0):
		for x in trial_set:
			print('trial_size1: ' + str(trial_set.size))	
			if x is not None:
				neighbours = get_neighbours(vertices, x.w, x.h, x.d, size)
				for y in neighbours:
					if y is not None:
						if y.state == 'FAR':
							y.parent = x
							y.state = 'TRIAL'
							trail_set = insert_trial_set(trial_set,y)
							print('trial_size2: ' + str(trial_set.size))	
						elif ((x.dt + x.edge_distance(max_intensity,x,y)) < y.dt):
							y.parent = x
							trail_set = extract_min_from_trial(trial_set)
							print('trial_size3: ' + str(trial_set.size))	
				break

			