import numpy as np

class spatial:
    def __init__(self, w, h, d):
        self.w = w
        self.h = h
        self.d = d
        self.parent = None
        self.index = None
        self.radius = 1

class segment_old():
	def __init__(self, leaf, root, length, level):
		self.leaf = leaf
		self.root = root
		self.length = length
		self.level = level
		self.parent = None

	def get_elements(self):
		if (self.leaf is None or self.root is None):
			print('incomplete segments')
			return

		out_swc = np.asarray([])
		# if (self.root_marker.parent is None):
		# print('root marker no parents')
		m = self.leaf
		# out_swc = np.append(out_swc,self.leaf)
		while (m != self.root):
			out_swc = np.append(out_swc, m)
			m = m.parent

		out_swc = np.append(out_swc,self.root)
		if (m != self.root):
			if (m is not None):
				print('error')
				print('m: ',m.w,m.h,m.d)
				print('leaf: ',self.leaf.w,self.leaf.h,self.leaf.d)
				print('root: ',self.root.w,self.root.h,self.root.d)

		return out_swc

class segment():
	def __init__(self, leaf, root, length, level):
		self.leaf = leaf
		self.root = root
		self.length = length
		self.level = level
		self.parent = None

	def get_elements(self):
		if(self.leaf is None):
			print('invalid segment')
			return
		result = np.array([])
		l = self.leaf
		while (l != self.root):
			result = np.append(result, l)
			l = l.parent

		result = np.append(result, l)

		return tuple(map(tuple,result))
