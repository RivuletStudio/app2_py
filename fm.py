import numpy as np
import numpy.linalg
import math
from utils.io import *
from marker import *

# strcuture for each voxel in the image
class vertex():
	def __init__(self,_w,_h,_d,_intensity,_state):
		self.w = _w
		self.h = _h
		self.d = _d
		self.intensity = _intensity
		self.state = _state
		self.prev = [_w,_h,_d]
		self.parent = [_w,_h,_d]
		self.phi = math.inf
		self.swc_index = 0

#  find average and max intensity
def find_max_intensity(img,size):
	max_intensity = 0
	total_intensity = 0
	not_zero = 0
	max_w = 0
	max_h = 0
	max_d = 0
	for w in range (size[0]):
		for h in range (size[1]):
			for d in range (size[2]):
				if img[w][h][d] > max_intensity:
					max_w = w
					max_h = h
					max_d = d
					max_intensity = img[w][h][d]
				# print(img[w][h][d])
	print('total intensity: ',total_intensity)
	print('max intensity: ',max_intensity,max_w,max_d,max_h)
	print('total vertices: ', count)
	return max_intensity

# find trial set
# def find_trial_set(vertices,max_w,max_h,max_d,size):
# 	# set source which has the largest dt value
# 	vertices[max_w][max_h][max_d].state = 'ALIVE'

# 	trial_set = np.array([],dtype=vertex)
# 	count = 0
	
# 	neighbours = get_neighbours(vertices,max_w,max_h,max_d,size) 

# 	# find trial set
# 	for i in neighbours:
# 		i.state = 'TRIAL'
# 		# set parent
# 		i.parent = vertices[max_w][max_h][max_d]
# 		# sort trial set
# 		trial_set = insert_trial_set(trial_set,i)
# 		count+=1
# 		print('Actual' + str(trial_set.size))
# 		print('Expected' + str(count))

# 	return trial_set

"""
Insert the vertex into trail_set 
Parameters
----------
trail_set : the numpy array which contains the all vertex with status TRAIL 
"""
def insert_trial_set(trial_set, vertex):
	size = trial_set.size

	if trial_set.size == 0:
		trial_set = np.append(trial_set,vertex)

	elif vertex.phi >= trial_set[size-1].phi:
		trial_set = np.insert(trial_set, size-1, vertex)

	else:
		index = 0
		for i in trial_set:
			if vertex.phi <= i.phi:
				trial_set = np.insert(trial_set,index,vertex)
				break
			index+=1

	return trial_set

"""
Extract the vertex with minimum distance to ALIVE set

Parameters
----------
trail_set : the numpy array which contains the all vertex with status TRAIL 
"""
def extract_min_from_trial(trial_set):
	min_vertex = trial_set[0]
	trial_set = np.delete(trial_set,0)
	print(trial_set.size)
	return trial_set,min_vertex

"""
Update the vertex distance and adjust its position in trail set

Parameters
----------
trail_set : the numpy array which contains the all vertex with status TRAIL
neighbour : the vertex needs to update the distance and adjust the position 
"""
def adjust_in_trial(trial_set,neighbour):
	index = 0
	if (trial_set.size == 0):
		return trial_set

	for i in trial_set:
		if neighbour.w == i.w and neighbour.h == i.h and neighbour.d == i.d:
			break
		index+=1
	trial_set = np.delete(trial_set,index)

	index = 0
	for i in trial_set:
		if neighbour.phi <= i.phi:
			trial_set = np.insert(trial_set,index,neighbour)
			break
		index+=1

	return trial_set

"""
Update the vertex distance and adjust its position in trail set

Parameters
----------
img : input img intensity stored in 3d numpy array
bimg : input binary intensity stored in 3d numpy array
size : input img size
max_w, max_h, max_d : the position where the seed location is
threshold : background/foreground threshold
allow_gap : if the fast marching needs to stride gaps
out_path : the out path to store the initial swc file 
"""
def fastmarching_dt_tree(img,bimg,size,max_w,max_h,max_d,threshold,allow_gap,out_path):

	max_intensity = numpy.amax(img)
	print('max intensity: ',max_intensity)

	# initialize a 3d numpy array to store all vertices
	vertices = np.empty((size[0],size[1],size[2]),dtype=vertex)

	# set all vertices have initial state FAR
	for w in range (size[0]):
		for h in range (size[1]):
			for d in range (size[2]):
				state = 'FAR'
				element = vertex(w, h, d, img[w][h][d], state)
				vertices[w][h][d] = element

	# set seed location to ALIVE
	vertices[max_w][max_h][max_d].state = 'ALIVE'
	vertices[max_w][max_h][max_d].phi = 0.0

	# initialize the trail_set and put seed location into trail set
	trial_set = np.array([],dtype=vertex)
	trial_set = insert_trial_set(trial_set,vertices[max_w][max_h][max_d])

	count = 0
	
	# extract the vertex with the minimum distance until the trail set is empty
	while(trial_set.size != 0):
		trial_set,min_vertex = extract_min_from_trial(trial_set)

		i = min_vertex.w
		j = min_vertex.h
		k = min_vertex.d

		min_ind = [i,j,k]
		prev_ind = min_vertex.prev

		# update the parent ndoe
		min_vertex.parent = min_vertex.prev

		min_vertex.state = 'ALIVE'

		# find all the neighbours of the vertex with the minimum distance
		for kk in range (-1,2):
			d = k+kk
			if(d < 0 or d >= size[2]):
				continue
			for jj in range (-1,2):
				h = j+jj
				if(h < 0 or h >= size[1]):
					continue
				for ii in range (-1,2):
					w = i+ii
					if(w < 0 or w >= size[0]):
						continue

					neighbour = vertices[w][h][d]

					if (allow_gap):
						if (img[w][h][d] <= threshold and img[i][j][k] <= threshold):
							continue
					else:
						if (img[w][h][d] <= threshold):
							continue

					# if the state of neighbour is not ALIVE, update the distance
					if neighbour.state != 'ALIVE':
						new_dist = min_vertex.phi + (1.0 - (img[w][h][d] - img[i][j][k]/max_intensity)*1000.0)
						prev_ind = min_ind

						# if the state of neighbour is FAR, insert into trail set
						if(neighbour.state == 'FAR'):
							neighbour.phi = new_dist
							neighbour.state = 'TRIAL'
							neighbour.prev = prev_ind
							trial_set = insert_trial_set(trial_set,neighbour)
						
						# if the state of neighbour is TRAIL, update the distance
						elif(neighbour.state == 'TRIAL'):
							if (neighbour.phi > new_dist):
								neighbour.phi = new_dist
								
								# adjust position in trail set
								trial_set = adjust_in_trial(trial_set,neighbour)
								neighbour.prev = prev_ind

		
	alive= []
	index = 1
	for w in range (size[0]):
		for h in range (size[1]):
			for d in range (size[2]):
				if vertices[w][h][d].state == 'ALIVE':
					vertices[w][h][d].swc_index = index
					# print('index: ',w,h,d)
					alive.append(vertices[w][h][d])
					index+=1
	print('total index: ',index)
	print('alive size: ',len(alive))
	
	out_tree = np.empty(len(alive), dtype=marker)
	print('--generate swc')

	swc = []
	for i in alive:
		p = i.parent
		ind = i.swc_index
		p_index = vertices[p[0]][p[1]][p[2]].swc_index
		swc.append([ind,3,i.w,i.h,i.d,1,p_index])
		new_marker = marker([i.w,i.h,i.d],ind,None,1)
		out_tree.itemset(ind-1,new_marker)
		# print(ind)

	seed_flag = 0
	seed_location = []
	seed_swc = 0
	for i in alive:
		p = i.parent
		ind = i.swc_index
		p_index = vertices[p[0]][p[1]][p[2]].swc_index
		if (ind == p_index):
			p_index = -1
		new_marker = marker([i.w,i.h,i.d],ind,out_tree[p_index-1],1)
		out_tree.itemset(ind-1,new_marker)

	for i in out_tree:
		if i.parent is None:
			print('ERROR!')


	ini_swc = np.asarray(swc)
	swc_x = ini_swc[:, 2].copy()
	swc_y = ini_swc[:, 3].copy()
	ini_swc[:, 2] = swc_y
	ini_swc[:, 3] = swc_x

	saveswc(out_path+'test.swc', ini_swc)
	print('swc size',len(swc))
	print('--generated swc saved at: ',out_path+'test.swc')
	# print('outtree size: ',out_tree.size)
	return out_tree





	


		