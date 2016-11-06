from node import *
from utils.io import *
import numpy as np
from scipy.spatial import distance


class segment():
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


def swc2topo_segs(alive, img, size):
	tol_num = alive.size

	leaf_nodes = np.array([])
	child_no = np.zeros(tol_num, dtype=int)

	for i in alive:
		if i.parent is None:
			continue
		else:
			# child_no[np.argwhere(alive == i.parent)] += 1
			child_no[i.parent.index] += 1

	index = 0
	for i in child_no:
		if i != 0:
			# loc = alive[index]
			leaf_nodes = np.append(leaf_nodes, alive[index])

		index += 1

	print('leaf_size: ', leaf_nodes.size)

	print('child no of index 31: ', child_no[8730])

	count = 0
	for i in alive:
		if i.parent is None:
			continue
		# elif np.argwhere(alive == i.parent) == 8370:
		elif i.parent.index == 8730:
			count += 1
	print('child no of index 31: ', count)

	leaf_num = leaf_nodes.size

	# calculate distance for every tree nodes (this part should have no errors)

	# furthest leaf distance for each tree node
	topo_dists = np.zeros(tol_num) 
	topo_leafs = np.empty(tol_num, dtype=spatial)

	for leaf in leaf_nodes:
		child_node = leaf
		parent_node = child_node.parent
		cid = child_node.index
		# cid = np.argwhere(alive == child_node)
		topo_leafs[cid] = leaf
		# topo_dists[cid] = img[leaf.w][leaf.h][leaf.d] / 255.0
		topo_dists[cid] = 0

		while (parent_node):
			print(childchild_node.w,)

			pid = parent_node.index
			# pid = np.argwhere(alive == parent_node)
			tmp_dst = img[parent_node.w][parent_node.h][
				parent_node.d] / 255.0 + topo_dists[cid]
			tmp_dst = distance.euclidean([parent_node.w,parent_node.h,parent_node.d],[child_node.w,child_node.h,child_node.d]) + topo_dists[cid]
			if (tmp_dst >= topo_dists[pid]):
				topo_dists[pid] = tmp_dst
				topo_leafs[pid] = topo_leafs[cid]
			else:
				break

			p = topo_leafs[pid]
			c = topo_leafs[cid]
			if leaf.w == 27 or leaf.h == 14 and leaf.d == 43:
				if p.w != 27 or p.h != 14 or p.d != 43:
					print('parent error',p.w,p.h,p.d)
					# print(tmp)
				
				if c.w != 27 or c.h != 14 or c.d != 43:
					print('error',c.w,c.h,c.d)


			child_node = parent_node
			cid = pid
			parent_node = parent_node.parent


	fp = np.argmax(topo_dists)
	fn = topo_leafs[fp]
	# fn2 = leaf_nodes[fp]
	print('furthest point location: ',fn.w,fn.h,fn.d,'length: ',topo_dists[fp])
	print('seed topo to this point: ',topo_leafs[0].w,topo_leafs[0].h,topo_leafs[0].d,topo_dists[0])


	# create Hierarchy Segments (this part has bugs)
	topo_segs = np.empty(leaf_num, dtype=segment)
	# topo_segs.fill(segment(0,0,1,0))

	f_index = 0
	for leaf in leaf_nodes:
		if leaf.w == 26 and leaf.h == 14 and leaf.d == 43:
			break
		f_index+=1
		# print(f_index)
		# print(leaf_nodes.size)
	leaf = leaf_nodes[f_index]
	root_parent = leaf.parent

	# while (root_parent):
	#   topo_leaf = topo_leafs[root_parent.index]
	#   root_marker = root_parent
	#   root_parent = root_marker.parent
	#   if topo_leaf == leaf:
	#       topo_leaf = topo_leafs[root_parent.index]
	#       print('topo leaf: ',topo_leaf.w,topo_leaf.h,topo_leaf.d)
	#       print('current: ',root_marker.w,root_marker.h,root_marker.d)

	#   else:
	#       print('stop: ',topo_leaf.w,topo_leaf.h,topo_leaf.d)
	#       print('current: ',root_marker.w,root_marker.h,root_marker.d)
	#       break


	# dst = topo_dists[root_marker.index]
	# print(root_marker.w,root_marker.h,root_marker.d)
	# print('DST: ',dst)



	index = 0
	for leaf in leaf_nodes:
	    root_marker = leaf
	    root_parent = root_marker.parent
	    level = 1

	    while (root_parent and topo_leafs[root_parent.index] == leaf):
	    # while (root_parent and topo_leafs[np.argwhere(alive == root_marker)] == leaf):
	        if child_no[root_marker.index] >= 2:
	        # if child_no[np.argwhere(alive == root_marker)] >= 2:
	            level += 1
	        root_marker = root_parent
	        root_parent = root_marker.parent


	    dst = topo_dists[root_marker.index]
	    # dst = topo_dists[np.argwhere(alive == root_marker)]

	    # if (leaf == root_marker and dst > 5):
	    #     print('NONONONO!')
	    #     print(dst)
	    topo_seg = segment(leaf, root_marker, dst, level)
	    topo_segs[index] = topo_seg

	    if (root_parent is None):
	        topo_seg.parent = 0
	    else:
	        leaf_marker2 = topo_leafs[root_parent.index]
	        # leaf_marker2 = topo_leafs[np.argwhere(alive == root_parent)]
	        topo_seg.parent = topo_segs[np.argwhere(leaf_nodes == leaf_marker2)]
	    index += 1



	# delete the segments less than length_threshold
	filter_segs = np.array([])

	for seg in topo_segs:
	    # seg_length = np.append(seg_length,seg.length)
	    if seg.length > 5:
	        filter_segs = np.append(filter_segs,seg)

	seg_swc = []

	count = 0
	for seg in filter_segs:
	  seg_tree = seg.get_elements()
	  for i in seg_tree:
	      if i.parent is None:
	          seg_swc.append([i.index,3,i.w,i.h,i.d,1,-1])
	          count+=1
	      else:
	          seg_swc.append([i.index,3,i.w,i.h,i.d,1,i.parent.index])
	seg_swc = np.asarray(seg_swc)
	swc_x = seg_swc[:, 2].copy()
	swc_y = seg_swc[:, 3].copy()
	seg_swc[:, 2] = swc_y
	seg_swc[:, 3] = swc_x
	saveswc('test/crop1/seg2.swc', seg_swc)

	# print(count)

	# # hierarchical pruning
	# visited_segs = np.array([])
	# sum_sig = sum_rdc = 0.0

	# # return topo_segs


	# return longest segment 
	# index = 0
	# max_len = 10
	# count = 0
	# for i in topo_segs:
		
	#   if i.length > max_len:
	#       max_len = index
	#       break
	#   index+=1
	#   count+=1

	# leaf = topo_segs[max_len]
	seg_swc = []


	index = 1
	# seg_tree = leaf.get_elements()
	seg_tree = np.array([])
	# print(type(leaf))
	# while (leaf.parent):
	#   seg_tree = np.append(seg_tree,leaf.get_elements())
	#   leaf = leaf.parent
	# seg_tree = np.append(seg_tree,leaf.get_elements())
	# print('longest length: ',leaf.length)
	# print('longest leaf: ',leaf.leaf.w,leaf.leaf.h,leaf.leaf.d)
	# print('longest root: ',leaf.root.w,leaf.root.h,leaf.root.d)
	# print('longest seg tree size: ',seg_tree.size)
	# for i in seg_tree:
	#   seg_swc.append([index,3,i.w,i.h,i.d,1,index+1])
	#   index+=1

	index = 0
	while(leaf.parent):
		leaf = leaf.parent
		seg_swc.append([index,3,leaf.w,leaf.h,leaf.d,1,index+1])
		index+=1
	seg_swc.append([index,3,leaf.w,leaf.h,leaf.d,1,index+1])

	seg_swc = np.asarray(seg_swc)
	swc_x = seg_swc[:, 2].copy()
	swc_y = seg_swc[:, 3].copy()
	seg_swc[:, 2] = swc_y
	seg_swc[:, 3] = swc_x
	saveswc('test/crop1/longest_seg2.swc', seg_swc)

	# for i in topo_segs:
	#   print('leaf',i.leaf.w,i.leaf.h,i.leaf.d)
	#   print('root',i.root.w,i.root.h,i.root.d)


# def topo_segs2swc(topo_segs):
#   # if 
#   if (topo_segs.size == 0)
#       return

#   min_dst = topo_segs[0].length
#   max_dst = min_dst
#   min_level = topo_segs[0].level
#   max_level = min_level

#   for seg in topo_segs:
#       dst = seg.length
#       min_dst = min(min_dst, dst)
#       max_dst = max(max_dst,dst)

#       level = seg.level
#       min_level = min(min_level, level)
#       max_level = max(max_level, level)

#   max_level = min(max_level,20)

#   max_dst -= min_dst
#   if (max_dst == 0.0):
#       max_dst = 0.0000001
#   max_level -= min_level
#   if (max_level == 0.0):
#       max_level =1

#   out
#   for seg in topo_segs:
#       dst = seg.length
#       level = min(seg.level, max_level)

#   return


def new_hp(alive, img, size):
	topo_segs = swc2topo_segs(alive, img, size)

	# filter_segs = np.array([])
	# for seg in topo_segs:
	#   if seg.length > 5:
	#       filter_segs = np.append(filter_segs,seg)

	# seg_swc = []

	# for seg in filter_segs:
	#   seg_tree = seg.get_segments()
	#   for i in seg_tree:
	#       seg_swc.append([i.index,3,i.w,i.h,i.d,1,i.parent.index])

	# seg_swc = np.asarray(seg_swc)
	# swc_x = seg_swc[:, 2].copy()
	# swc_y = seg_swc[:, 3].copy()
	# seg_swc[:, 2] = swc_y
	# seg_swc[:, 3] = swc_x
	# saveswc('test/crop2/seg.swc', seg_swc)

	# topo_segs2swc(filter_segs)
