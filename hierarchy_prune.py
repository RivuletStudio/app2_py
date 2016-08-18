import numpy as np

class HierarchySegment():
	def __init__(self,_leaf=0,_root=0,_len=0,_level=1,_parent=0):
		self.leaf_marker = _leaf
		
		# its parent marker is in current segment's parent segment 
		self.root_marker = _root

		# the length from leaf to root
		self.length = _len
		
		# the segments number from leaf to root
		self.level = _level
		self.parent = _parent

	def get_markers(outswc):
		if (not leaf_marker || not root_marker):
			return

		p = leaf_marker
		while (p != root_marker):
			outswc.append(p)
