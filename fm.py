import numpy as np
import numpy.linalg
import math


class vertex():
	def __init__(self,_ind,_w,_h,_d,_dt,_intensity,_state):
		self.ind = _ind
		self.w = _w
		self.h = _h
		self.d = _d
		self.dt= _dt
		self.intensity = _intensity
		self.state = _state
		self.neighbours = []

	def add_neighbours(vertex_a,vertex_b):
		vertex_a.neighbours.append(vertex_b)
		vertex_b.neighbours.append(vertex_a)

	def euc_distance(self,vertex_a,vertex_b):
		distance = np.linalg.norm(vertex_a-vertex_b)
		return distance

	def geodesic_distance(self,intensity_max,_vertex):
		# this 10 can be set up as a parameter
		return math.exp(10 * ((1-_vertex.intensity/intensity_max) ** 2))


# initialize the fast marching tree
def initialize(size,img,dtimg,bimg):
	vertices = []
	index = 0
	for w in range (size[0]):
		for h in range (size[1]):
			for d in range (size[2]):
				flag = 'FAR'
				if bimg[i][j][k] == 1:
					flag = 'ALIVE'
				element = vertex(index, w, h, d, dtimg[i][j][k], img[i][j][k], flag)
				vertices.append(element)

	for i in np.where(bimg == 1):
		print('--soma location')
		print(i)
			
	return vertices

def get_neighbours(vertices,vertex,size):
	neighbours = []
	w = vertex.w
	h = vertex.h
	d = vertex.d

	if (x-1 >= 0):
		vertex.neighbours.append(vertices[w*size[0]+h*size[1]+d*size[2]])
	if (x+1 <= size[0]-1):
		vertex.neighbours.append



def set_trial_set(vertices):
	for i in vertices:
		if i.state == 'ALIVE':
			pass