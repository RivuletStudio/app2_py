import numpy as np 
class marker():
	def __init__(self,spatial_index,swc_index,parent):
		self.w = spatial_index[0]
		self.h = spatial_index[1]
		self.d = spatial_index[2]
		self.swc_index = swc_index
		self.index_map = None
		self.parent = parent
		self.child_no = 0
		self.radius = 1
		self.type = 3

