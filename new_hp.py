from node import *
from utils.io import *
import numpy as np
from scipy.spatial import distance

class segment():
	def __init__(self,leaf,root,length,level):
		self.leaf = leaf
		self.root = root
		self.length = length
		self.level = level
		self.parent = None

	def get_segments(self):
		if (self.leaf is None or self.root is None):
			print('incomplete segments')
			return

		out_swc = np.asarray([])
		# if (self.root_marker.parent is None):
			# print('root marker no parents')
		m = self.leaf
		out_swc = np.append(out_swc,m)
		while (m.parent):
			out_swc = np.append(out_swc,m)
			m = m.parent

		# out_swc = np.append(out_swc,self.root)

		return out_swc

def swc2topo_segs(alive,img,size):
	tol_num = alive.size
	
	leaf_nodes = np.array([])
	child_no = np.zeros(tol_num,dtype=int)

	for i in alive:
		if i.parent.index == -1:
			continue
		else:
			child_no[i.parent.index-1]+=1

	index = 0
	for i in child_no:
		if i != 0:
			# loc = alive[index]
			leaf_nodes = np.append(leaf_nodes,alive[index])

		index+=1

	# test_swc = []
	# index = 1
	# for i in leaf_nodes:
	# 	test_swc.append([index,3,i.w,i.h,i.d,1,index+1])
	# 	index+=1

	# test_swc = np.asarray(test_swc)
	# swc_x = test_swc[:, 2].copy()
	# swc_y = test_swc[:, 3].copy()
	# test_swc[:, 2] = swc_y
	# test_swc[:, 3] = swc_x
	# saveswc('test/crop2/leaf.swc', test_swc)

	print('leaf_size: ',leaf_nodes.size)

	print('child no of index 31: ',child_no[8730])

	count = 0
	for i in alive:
		if i.parent.index == -1:
			continue
		elif i.parent.index == 8731:
			count+=1
	print('child no of index 31: ',count)

	
	leaf_num = leaf_nodes.size

	# calculate distance for every nodes
	topo_dists = np.zeros(tol_num)
	topo_leafs = np.empty(tol_num,dtype=spatial)

	for leaf in leaf_nodes:
		child_node = leaf
		parent_node = child_node.parent
		cid = child_node.index-1
		topo_leafs[cid] = leaf
		topo_dists[cid] = img[leaf.w][leaf.h][leaf.d] / 255.0
		# topo_dists[cid] = 0

		while(parent_node):
			# print(parent_node.index,parent_node.w,parent_node.h,parent_node.d)
			pid = parent_node.index-1
			tmp_dst = img[parent_node.w][parent_node.h][parent_node.d]/255.0 + topo_dists[cid]
			# tmp_dst = distance.euclidean([parent_node.w,parent_node.h,parent_node.d],[child_node.w,child_node.h,child_node.d]) + topo_dists[cid]
			if (tmp_dst >= topo_dists[pid]):
				topo_dists[pid] = tmp_dst
				topo_leafs[pid] = topo_leafs[cid]
			else:
				break
			child_node = parent_node
			cid = pid
			parent_node = parent_node.parent

	# create Hierarchy Segments
	topo_segs = np.empty(leaf_num,dtype=segment)
	leaf_ind_map = np.empty(leaf_num,dtype=spatial)
	index = 0
	for i in leaf_nodes:
		leaf_ind_map[index] = i
		index+=1

	index = 0
	for leaf in leaf_nodes:
		root_marker = leaf
		root_parent = root_marker.parent
		level = 1

		while (root_parent and topo_leafs[root_parent.index-1] == leaf):
			if child_no[root_marker.index-1] >= 2:
				level+=1
			root_marker = root_parent
			root_parent = root_marker.parent

		dst = topo_dists[root_marker.index-1]

		topo_seg = segment(leaf,root_marker,dst,level)
		topo_segs[index] = topo_seg
		if (root_parent is None):
			topo_seg.parent = 0
		else:
			leaf_marker2 = topo_leafs[root_parent.index-1]
			leaf_ind2 = 0
			for i in leaf_ind_map:
				# print(i)
				if i.w == leaf_marker2.w and i.h == leaf_marker2.h and i.d == leaf_marker2.d:
					break;
				leaf_ind2+=1
			topo_seg.parent = topo_segs[leaf_ind2]	 

		index+=1

	# index = 0
	# max_len = 0
	# for i in topo_segs:
	# 	if i.length > max_len:
	# 		max_len = index
	# 	index+=1

	# leaf = topo_segs[max_len]
	# seg_swc = []

	# index = 1
	# while leaf.parent:
	# 	seg_swc.append([index,3,leaf.w,leaf.h,leaf.d,1,index+1])
	# 	leaf = leaf.parent
	# 	index+=1

	# index = 1
	# seg_tree = leaf.get_segments()
	# for i in seg_tree:
	# 	seg_swc.append([index,3,i.w,i.h,i.d,1,index+1])
	# 	index+=1

	# seg_swc = np.asarray(seg_swc)
	# swc_x = seg_swc[:, 2].copy()
	# swc_y = seg_swc[:, 3].copy()
	# seg_swc[:, 2] = swc_y
	# seg_swc[:, 3] = swc_x
	# saveswc('test/crop2/seg.swc', seg_swc)

	# for i in topo_segs:
	# 	print('leaf',i.leaf.w,i.leaf.h,i.leaf.d)
	# 	print('root',i.root.w,i.root.h,i.root.d)


	return topo_segs

# def topo_segs2swc(topo_segs):
# 	# if 
# 	if (topo_segs.size == 0)
# 		return

# 	min_dst = topo_segs[0].length
# 	max_dst = min_dst
# 	min_level = topo_segs[0].level
# 	max_level = min_level

# 	for seg in topo_segs:
# 		dst = seg.length
# 		min_dst = min(min_dst, dst)
# 		max_dst = max(max_dst,dst)

# 		level = seg.level
# 		min_level = min(min_level, level)
# 		max_level = max(max_level, level)

# 	max_level = min(max_level,20)

# 	max_dst -= min_dst
# 	if (max_dst == 0.0):
# 		max_dst = 0.0000001
# 	max_level -= min_level
# 	if (max_level == 0.0):
# 		max_level =1

# 	out
# 	for seg in topo_segs:
# 		dst = seg.length
# 		level = min(seg.level, max_level)


# 	return

def new_hp(alive,img,size):
	topo_segs = swc2topo_segs(alive,img,size)

	filter_segs = np.array([])
	for seg in topo_segs:
		if seg.length > 5:
			filter_segs = np.append(filter_segs,seg)

	seg_swc = []

	for seg in filter_segs:
		seg_tree = seg.get_segments()
		for i in seg_tree:
			seg_swc.append([i.index,3,i.w,i.h,i.d,1,i.parent.index])

	seg_swc = np.asarray(seg_swc)
	swc_x = seg_swc[:, 2].copy()
	swc_y = seg_swc[:, 3].copy()
	seg_swc[:, 2] = swc_y
	seg_swc[:, 3] = swc_x
	saveswc('test/crop2/seg.swc', seg_swc)


	# topo_segs2swc(filter_segs)

			


