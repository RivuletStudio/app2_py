import numpy as np
import math
from utils.io import *
from marker import *
from scipy.spatial import distance

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
		# if (self.root_marker.parent is None):
			# print('root marker no parents')
		m = self.leaf_marker
		while (m != self.root_marker):
			out_swc = np.append(out_swc,m)
			m = m.parent

		out_swc = np.append(out_swc,self.root_marker)

		return out_swc


def swc2topo_segs(ini_swc,img,size):
	# Phase1: calculate distance for every nodes

	leaf_markers = []
	tol_num = ini_swc.size

	index = 0
	for i in ini_swc:
		i.swc_map = index
		index+=1

	# count the number of children of each node in initial swc marker
	for i in ini_swc:
		if i.parent is None:
			print('Node in initial swc has no parent!')
			continue
		else:
			ini_swc[i.parent.swc_map].child_no +=1

	index = 0
	for i in ini_swc:
		if i.child_no == 0:
			leaf_markers.append(i)

	leaf_markers = np.asarray(leaf_markers)
	leaf_num = leaf_markers.size
	print('number of leaves: ',leaf_markers.size)

	# furthest leaf distance for each tree node
	topo_dists = np.empty(tol_num, dtype=float)
	topo_leaves = np.empty(tol_num, dtype=marker)

	index = 0
	for leaf in leaf_markers:
		parent_node = leaf.parent
		child_node = leaf
		cid = child_node.swc_index-1
		topo_leaves.itemset(cid,leaf)
		# leaf.index_map = index

		# internsity distance method
		# tmp_topo_dst = img[leaf.w][leaf.h][leaf.d]/255.0
		tmp_topo_dst = 0
		topo_dists.itemset(cid,tmp_topo_dst)

		while(parent_node):
			pid = parent_node.swc_index-1
			# tmp_dst = img[parent_node.w][parent_node.h][parent_node.d]/255.0 + topo_dists[cid]
			tmp_dst = distance.euclidean([parent_node.w,parent_node.h,parent_node.d],[child_node.w,child_node.h,child_node.d]) + topo_dists[cid]
			if (tmp_dst >= topo_dists[pid]):
				# topo_dists[pid] = tmp_dst
				# topo_leaves[pid] = topo_leaves[cid]
				print(tmp_dst)
				topo_dists.itemset(pid,tmp_dst)
				topo_leaves.itemset(pid,topo_leaves.item(cid))
			else:
				break
			child_node = parent_node
			cid = pid
			parent_node = parent_node.parent

		index+=1

	# print(topo_dists)
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

	
	# Phase2: create hierarchy segments
	topo_segs = np.empty(leaf_num,dtype=segment)

	index = 0
	for leaf in leaf_markers:
		leaf.leaf_index_map = index
		index+=1

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
			leaf_ind2 = leaf_marker2.leaf_index_map
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

def marker_radius(img,out_marker,size,threshold):
	max_r = min(size[0]/2,size[1]/2,size[2]/2)

	ir = 1
	while ir < int(max_r+1):
		total_num = 0
		back_num = 0

		zlower = -ir
		zupper = ir
		dz = zlower
		dy = -ir
		dx = -ir
		while dz <= zupper:
			dz += dz
			while dy <= ir:
				dy += dy
				while dx <= ir:
					dx += dx
					total_num+=1

					# print('ir:' ,ir)
					r = math.sqrt(dx*dx + dy*dy + dz*dz)
					# print(r)
					if (r>ir-1 or r<=ir):
						w = out_marker.w+dx
						if (w<0 or w>=size[0]):
							return ir

						h = out_marker.h+dy
						if (h<0 or h>=size[1]):
							return ir

						d = out_marker.d+dz
						if (d<0 or d>=size[2]):
							return ir

						if (img[w][h][d] <= threshold):
							back_num+=1

							if((back_num/total_num) > 0.001):
								return ir
					# print('dx: ',dx,'dy: ',dy,'dz: ',dz)

		ir += 1
	return ir



def hierarchy_prune(ini_swc,img,size,out_path,threshold):
	topo_segs = swc2topo_segs(ini_swc,img,None)
	print('topo size: ',topo_segs.size)
	filter_segs = np.asarray([])
	for seg in topo_segs:
		# this 10 can be set as parameter??????????
		# print(seg.length)
		if (seg.length > 3):
			filter_segs = np.append(filter_segs,seg)


	# smooth_curve 
	# seg_markers = np.asarray([])
	# for seg in filter_segs:
	# 	leaf_marker = seg.leaf_marker
	# 	root_marker = seg.root_marker
	# 	p = leaf_marker
	# 	while (p != root_marker):
	# 		seg_markers = np.append(seg_markers,p)
	# 		p = p.parent
	# 	seg_markers = np.append(seg_markers,root_marker)
	# filter_segs = smooth_curve_and_radius(seg_markers)

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
		radius = marker_radius(img,out,size,threshold)
		if out.parent is None:
			swc.append([out.index_map,out.type,out.w,out.h,out.d,radius,-1])
		else:
			swc.append([out.index_map,out.type,out.w,out.h,out.d,radius,out.parent.index_map])

		if out is None:
			print('out is None!')
	f_swc = np.asarray(swc)

	swc_x = f_swc[:, 2].copy()
	swc_y = f_swc[:, 3].copy()
	f_swc[:, 2] = swc_y
	f_swc[:, 3] = swc_x

	index = 0
	for i in f_swc[:,6]:
		if i is None:
			f_swc[index][6] = -1
			print('Error 6')
		index+=1

	# print(f_swc[:,0])
	saveswc(out_path+'my_final_smooth.swc',f_swc)









# def happ(ini_swc,img,size,out_path):
# 	leaf_markers = []
# 	tol_num = ini_swc.size
# 	count = 0
# 	for i in ini_swc:
# 		if i.parent is None:
# 			count+=1
# 			continue
# 		else:
# 			ini_swc[i.parent.swc_index-1].child_no +=1

# 	index = 0
# 	for i in ini_swc:
# 		if i.child_no == 0:
# 			leaf_markers.append(i)

# 	leaf_markers = np.asarray(leaf_markers)
# 	topo_segs = swc2topo_segs(ini_swc,img,size)

# 	filter_segs = np.asarray([])




# def smooth_curve_and_radius(seg_markers):
# 	# n = seg_markers.size
# 	halfwin = 2.5
# 	mc = seg_markers
# 	index = 1
# 	for seg in mc:
# 		winC = []
# 		winW = []
# 		winC.append(seg)
# 		winW.append(1.0+halfwin)

# 		for j in range(3):
# 			k1 = index + j
# 			k2 = index - j
# 			winC.append(mc.item(k1)) 
# 			winC.append(mc.item(k2))
# 			winW.append(1.0+halfwin-j)
# 			winW.append(1.0+halfwin-j)

# 		s=w=h=d=r=0.0
# 		k=0
# 		while (k < len(winC)):
# 			w += winW[k] * winC[k].w
# 			h += winW[k] * winC[k].h
# 			d += winW[k] * winC[k].d
# 			r += winW[k] * winC[k].radius
# 			k+=1

# 		if (s):
# 			w = w/s
# 			h = h/s
# 			d = d/s
# 			r = r/s

# 		update_marker = None
# 		if seg.parent is None:
# 			update_marker = marker([seg.w,seg.h,seg.d],seg.swc_index,0,r)
# 		else :
# 			update_marker = marker([seg.w,seg.h,seg.d],seg.swc_index,seg.parent,r)
# 		seg_markers.itemset(index,update_marker)


# 		index +=1
# 		if (index == mc.size-2):
# 			break
# 	return seg_markers