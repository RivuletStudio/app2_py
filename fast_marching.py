import numpy as np
import numpy.linalg
import math


class vertex():
	def __init__(self,_ind,_x,_y,_z,_dt,_intensity,_state):
		self.ind = _ind
		self.x = _x
		self.y = _y
		self.z = _z
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

	def geodesic_distance(self,intensity_max,vertex):
		# this 10 can be set up as a parameter
		return math.exp(10 * ((1-vetex.intensity/intensity_max) ** 2))



