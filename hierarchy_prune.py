import numpy as np
import math
from utils.io import *
from marker import *

class segment():
	def __init__(self,leaf_marker,root_marker,length,level):
		self.leaf_marker = leaf_marker
		self.root_marker = root_marker
		self.length = length
		self.level = level

	def get_markers(self):
		if (self.leaf_marker is None or self.root_marker is None):
			print('incomplete segments')
			return

		out_swc = np.asarray([])
		if (self.root_marker.parent is None):
			print('root marker no parents')
		m = self.leaf_marker
		while (m != self.root_marker):
			out_swc = np.append(out_swc,m)
			m = m.parent

		out_swc = np.append(out_swc,self.root_marker)

		return out_swc


def swc2topo_segs(ini_swc,img,size):
	# calculate distance for every nodes

	leaf_markers = []
	tol_num = ini_swc.size
	count = 0
	for i in ini_swc:
		if i.parent is None:
			count+=1
			continue
		else:
			ini_swc[i.parent.swc_index-1].child_no +=1

	index = 0
	for i in ini_swc:
		if i.child_no == 0:
			leaf_markers.append(i)

	leaf_markers = np.asarray(leaf_markers)
	leaf_num = leaf_markers.size
	print('number of leaves: ',leaf_markers.size)
	print('number of orphan: ',count)

	topo_dists = np.empty(tol_num, dtype=float)
	topo_leaves = np.empty(tol_num, dtype=marker)
	leaf_ind_map = []
	index = 0
	for leaf in leaf_markers:
		parent_node = leaf.parent
		cid = leaf.swc_index-1
		topo_leaves.itemset(cid,leaf)
		leaf.index_map = index
		# internsity distance method
		# print(leaf.w,leaf.h,leaf.d)
		tmp_topo_dst = img[leaf.w][leaf.h][leaf.d]/255.0
		topo_dists.itemset(cid,tmp_topo_dst)

		while(parent_node):
			pid = parent_node.swc_index-1
			tmp_dst = img[parent_node.w][parent_node.h][parent_node.d]/255.0 + topo_dists[cid]
			if (tmp_dst >= topo_dists[pid]):
				topo_dists[pid] = tmp_dst
				topo_leaves[pid] = topo_leaves[cid]
			else:
				break
			child_node = parent_node
			cid = pid
			parent_node = parent_node.parent

		index+=1

	count = 0
	for i in topo_leaves:
		if i is not None:
			count+=1

	print('topoleaves: ',count)

	count = 0
	for i in topo_dists:
		if i is not None:
			count+=1

	print('topodists: ',count)

	topo_segs = np.empty(leaf_num,dtype=segment)

	index = 0
	for leaf in leaf_markers:
		root_marker = leaf
		root_parent = root_marker.parent
		level = 1
		while(root_parent and topo_leaves[root_parent.swc_index-1] == leaf):
			if(root_marker.child_no >= 2):
				level+=1
			root_marker = root_parent
			root_parent = root_marker.parent

		dst = topo_dists[root_marker.swc_index-1]

		topo_seg = segment(leaf, root_marker, dst, level)
		topo_segs.itemset(index, topo_seg)

		if (root_parent is None):
			topo_seg.parent = 0
		else:
			leaf_marker2 = topo_leaves[root_parent.swc_index-1]
			leaf_ind2 = leaf_marker2.index_map
			topo_seg.parent = topo_segs[leaf_ind2]
		index+=1

	return topo_segs

def topo_seg2swc(topo_segs):
	if (topo_segs.size == 0):
		print('segments are empty!')
		return

	min_dst = topo_segs[0].length
	max_dst = min_dst
	min_level = topo_segs[0].level
	max_level = min_level
	for seg in topo_segs:
		dst = seg.length
		min_dst = min(min_dst, dst)
		max_dst = max(max_dst, dst)

		level = seg.level
		min_level = min(min_level,level)
		max_level = max(max_level,level)


	max_level = min(max_level, 20)
	print('min dst: ',min_dst)
	print('max dst: ',max_dst)
	print('min_level',min_level)
	print('max_level',max_level)

	max_dst -= min_dst
	if (max_dst == 0.0):
		max_dst = 0.0000001
	max_level -= min_level 
	if (max_level == 0):
		max_level = 1.0

	out_markers = np.asarray([])
	for seg in topo_segs:
		dst = seg.length
		level = min(seg.level, max_level)
		color_id = (level - min_level)/max_level * 254.0 + 20.5
		tmp_markers = seg.get_markers()
		for tmp in tmp_markers:
			tmp.type = color_id
		out_markers = np.append(out_markers,tmp_markers)

	return out_markers


def hierarchy_prune(ini_swc,img,out_path):
	topo_segs = swc2topo_segs(ini_swc,img,None)
	print('topo size: ',topo_segs.size)
	filter_segs = np.asarray([])
	for seg in topo_segs:
		# this 10 can be set as parameter??????????
		# print(seg.length)
		if (seg.length > 0.1):
			filter_segs = np.append(filter_segs,seg)

	print('filter topo size: ',filter_segs.size)
	out_markers = topo_seg2swc(filter_segs)
	print('out size: ',out_markers.size)

	count = 0
	for out in out_markers:
		if out.parent is None:
			count+=1
			print('out: ',out.swc_index,0)
		# else:
			# print('out: ',out.swc_index,out.parent.swc_index)
	print('none in out_markers: ',count)

	swc = []
	index = 1
	for out in out_markers:
		out.index_map = index
		index+=1

	for out in out_markers:
		if out.parent is None:
			swc.append([out.index_map,out.type,out.w,out.h,out.d,out.radius,-1])
		else:
			swc.append([out.index_map,out.type,out.w,out.h,out.d,out.radius,out.parent.index_map])

		if out is None:
			print('out is None!')
	f_swc = np.asarray(swc)

	swc_x = f_swc[:, 2].copy()
	swc_y = f_swc[:, 3].copy()
	f_swc[:, 2] = swc_y
	f_swc[:, 3] = swc_x

	for i in f_swc[:,0]:
		if i is None:
			print('Error 0')

	for i in f_swc[:,1]:
		if i is None:
			print('Error 0')

	index = 0
	for i in f_swc[:,6]:
		if i is None:
			f_swc[index][6] = -1
			print('Error 6')
		index+=1

	# print(f_swc[:,0])
	saveswc(out_path+'my_final.swc',f_swc)

def happ()