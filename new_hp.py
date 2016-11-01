class segment():
	def __init__(self,leaf,root,length,level):
		self.leaf = leaf
		self.root = root
		self.length = length
		self.level = level

	def get_segments(self):
		if (self.leaf is None or self.root is None):
			print('incomplete segments')
			return

		out_swc = np.asarray([])
		# if (self.root_marker.parent is None):
			# print('root marker no parents')
		m = self.leaf_marker
		while (m != self.root):
			out_swc = np.append(out_swc,m)
			m = m.parent

		out_swc = np.append(out_swc,self.root)

		return out_swc

def swc2topo_segs(ini_swc,img,size):
	tol_num = ini_swc.size
	
	leaf_nodes = np.empty(tol_num,dtype=spatial)
	child_no = np.zeros(tol_num,dtype=int)

	


def new_hp(ini_swc,img,size):
	swc2topo_segs(ini_swc,img,size)

