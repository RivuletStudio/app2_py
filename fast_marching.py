import numpy as np
import numpy.linalg


class vertex():
	def __init__(self,_ind,_x,_y,_z,_distance,_state):
		self.ind = _ind
		self.x = _x
		self.y = _y
		self.z = _z
		self.distance = _distance
		self.state = _state

	def euc_distance(vertex_a,vertex_b):
		distance = np.linalg.norm(vertex_a-vertex_b)
		return distance

	def geodesic_distance()


