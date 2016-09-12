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
		self.neighbours = []

	def add_neighbours(self,vertex):
		self.neighbours.append(vertex)
		vertex.neighbours.append(self)

	def euc_distance(self,vertex):
		distance = math.sqrt((self.w-vertex.w) ** 2 + (self.h-vertex.h) ** 2 + (self.d-vertex.d) ** 2)
		return distance

	def geodesic_distance(self,intensity_max,vertex):
		# this 10 can be set up as a parameter
		return math.exp(10 * ((1-vertex.intensity/intensity_max) ** 2))

	# def edge_distance(self,)


# initialize the fast marching tree
def initialize(size,img,dtimg,bimg):
	vertices = np.empty((size[0],size[1],size[2]),dtype=vertex)
	for w in range (size[0]):
		for h in range (size[1]):
			for d in range (size[2]):
				state = 'FAR'
				# print(index)
				if bimg[w][h][d] == 1:
					state = 'ALIVE'
				element = vertex(w, h, d, dtimg[w][h][d], img[w][h][d], state)
				vertices[w][h][d] = element


	return vertices

def find_trial_set(vertices,size):
	trial_set = []
	count = 0
	for w in range (size[0]):
		for h in range (size[1]):
			for d in range (size[2]):
				if vertices[w][h][d].state == 'ALIVE':
					neighbours = get_neighbours(vertices,w,h,d,size) 
					for i in neighbours:
						if i.state == 'FAR':
							i.state = 'TRIAL'
							trial_set.append(i)
							count+=1
	print(count)
	return trial_set

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
