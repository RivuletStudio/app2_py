import numpy as np
import numpy.linalg
import math
from utils.io import *
from node import *
from heap import *

givals = [22026.5,   20368, 18840.3, 17432.5, 16134.8, 14938.4, 13834.9, 12816.8, 
11877.4, 11010.2, 10209.4,  9469.8, 8786.47, 8154.96, 7571.17, 7031.33, 
6531.99, 6069.98, 5642.39, 5246.52, 4879.94, 4540.36, 4225.71, 3934.08, 
 3663.7, 3412.95, 3180.34,  2964.5, 2764.16, 2578.14, 2405.39,  2244.9, 
2095.77, 1957.14, 1828.24, 1708.36, 1596.83, 1493.05, 1396.43, 1306.47, 
1222.68, 1144.62, 1071.87, 1004.06, 940.819, 881.837, 826.806, 775.448, 
727.504, 682.734, 640.916, 601.845, 565.329, 531.193, 499.271, 469.412, 
441.474, 415.327, 390.848, 367.926, 346.454, 326.336, 307.481, 289.804, 
273.227, 257.678, 243.089, 229.396, 216.541, 204.469, 193.129, 182.475, 
172.461, 163.047, 154.195, 145.868, 138.033, 130.659, 123.717, 117.179, 
111.022,  105.22, 99.7524, 94.5979, 89.7372, 85.1526,  80.827, 76.7447, 
 72.891, 69.2522, 65.8152, 62.5681, 59.4994, 56.5987,  53.856, 51.2619, 
48.8078, 46.4854, 44.2872, 42.2059, 40.2348, 38.3676, 36.5982, 34.9212, 
33.3313, 31.8236, 30.3934, 29.0364, 27.7485,  26.526,  25.365, 24.2624, 
23.2148, 22.2193,  21.273, 20.3733, 19.5176, 18.7037, 17.9292,  17.192, 
16.4902,  15.822, 15.1855,  14.579, 14.0011, 13.4503, 12.9251, 12.4242, 
11.9464, 11.4905, 11.0554, 10.6401, 10.2435, 9.86473, 9.50289, 9.15713, 
8.82667, 8.51075, 8.20867, 7.91974, 7.64333, 7.37884, 7.12569, 6.88334, 
6.65128, 6.42902,  6.2161, 6.01209, 5.81655, 5.62911, 5.44938, 5.27701, 
5.11167, 4.95303, 4.80079, 4.65467, 4.51437, 4.37966, 4.25027, 4.12597, 
4.00654, 3.89176, 3.78144, 3.67537, 3.57337, 3.47528, 3.38092, 3.29013, 
3.20276, 3.11868, 3.03773,  2.9598, 2.88475, 2.81247, 2.74285, 2.67577, 
2.61113, 2.54884, 2.48881, 2.43093, 2.37513, 2.32132, 2.26944, 2.21939, 
2.17111, 2.12454, 2.07961, 2.03625, 1.99441, 1.95403, 1.91506, 1.87744, 
1.84113, 1.80608, 1.77223, 1.73956, 1.70802, 1.67756, 1.64815, 1.61976, 
1.59234, 1.56587, 1.54032, 1.51564, 1.49182, 1.46883, 1.44664, 1.42522, 
1.40455,  1.3846, 1.36536,  1.3468,  1.3289, 1.31164, 1.29501, 1.27898, 
1.26353, 1.24866, 1.23434, 1.22056,  1.2073, 1.19456, 1.18231, 1.17055, 
1.15927, 1.14844, 1.13807, 1.12814, 1.11864, 1.10956, 1.10089, 1.09262, 
1.08475, 1.07727, 1.07017, 1.06345, 1.05709, 1.05109, 1.04545, 1.04015, 
1.03521,  1.0306, 1.02633, 1.02239, 1.01878,  1.0155, 1.01253, 1.00989, 
1.00756, 1.00555, 1.00385, 1.00246, 1.00139, 1.00062, 1.00015,       1]

def GI(index,img,max_intensity,min_intensity):
	return givals[(int)((img[index.w][index.h][index.d] - min_intensity)/max_intensity * 255)]

#  find average and max intensity
def find_max_intensity(img,size):
	max_intensity = 0
	min_intensity = np.inf
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

def insert(trail_set, phi, new_dist, spatial):
	ind = 0
	if trail_set is None:
		trail_set =  np.insert(trail_set,ind,spatial)
		# print('after insert: ',trail_set.size)
		return trail_set

	print('size: ',trail_set.size)
	for i in trail_set:
		if new_dist < phi[i.w][i.h][i.d]:
			trail_set =  np.insert(trail_set,ind,spatial)
			# print('after insert: ',trail_set.size)
			return trail_set
		ind+=1
	trail_set =  np.insert(trail_set,ind,spatial)
	return trail_set

def find_adjust(trail_set, phi, new_dist, spatial):
	index = 0
	for i in trail_set:
		if (i.w == spatial.w and i.h == spatial.h and i.d == spatial.d):
			break
		index+=1


	trail_set = np.delete(trail_set,index)
	
	ind = 0

	for i in trail_set:
		if new_dist < phi[i.w][i.h][i.d]:
			trail_set =  np.insert(trail_set,ind,spatial)
			return trail_set,ind
		ind+=1
	trail_set =  np.insert(trail_set,ind,spatial)
	# print('after insert: ',trail_set.size)
	return



def fastmarching_dt_tree(img,bimg,size,seed_w,seed_h,seed_d,threshold,allow_gap,out_path):
	max_intensity = np.amax(img)
	print('max internsity:', max_intensity)


	# state 0 for FAR, state 1 for TRAIL, state 2 for ALIVE
	state = np.zeros((size[0],size[1],size[2]))

	# initialize 
	phi = np.empty((size[0],size[1],size[2]),dtype = np.float32)
	parent = np.empty((size[0],size[1],size[2]),dtype = spatial)
	prev = np.zeros((size[0],size[1],size[2]),dtype = spatial)
	# trail_set = np.array([])
	# trail_index = np.zeros((size[0],size[1],size[2]),dtype = np.int32)

	for w in range(size[0]):
		for h in range(size[1]):
			for d in range(size[2]):
				parent[w][h][d] = spatial(w,h,d)
				phi[w][h][d] = np.inf

	# put seed into ALIVE set
	state[seed_w][seed_h][seed_d] = 2
	phi[seed_w][seed_h][seed_d] = 0.0


	spatial_index = spatial(seed_w,seed_h,seed_d)
	trail_set = np.asarray([spatial_index])
	# print('11111size: ',trail_set.size)
	index = 0

	while (trail_set.size != 0):
		# print('size: ',trail_set.size)
		min_ind = trail_set.item(0)
		trail_set = np.delete(trail_set,0)
		print('size: ',trail_set.size)
		# print('after extract: ',trail_set.size)
		# min_ind = min_elem.index
		# print(min_ind)
		i = min_ind.w
		j = min_ind.h
		k = min_ind.d
		prev_ind = prev[i][j][k]

		parent[i][j][k] = prev_ind

		state[i][j][k] = 2

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

					offset = abs(ii) + abs(jj) + abs(kk)
					# print('offset: ',offset)
					# this 2 is cnn type
					if offset == 0 or offset > 2:
						continue

					factor = 1
					if offset == 2:
						factor = 1.414214
					elif offset == 3:
						factor = 1.732051

					if (allow_gap):
						if (img[w][h][d] <= threshold and img[i][j][k] <= threshold):
							continue
					else:
						if (img[w][h][d] <= threshold):
							continue

					spatial_index = spatial(w,h,d)
					if (state[w][h][d] != 2):
						# min_intensity set as 0
						new_dist = phi[w][h][d] + (GI(spatial_index,img,max_intensity,0.0) + GI(min_ind,img,max_intensity,0.0))*factor*0.5
						prev_ind = min_ind

						if(state[w][h][d] == 0):
							phi[w][h][d] = new_dist
							# spatial_index = spatial(w,h,d)
							trail_set = insert(trail_set,phi,new_dist,spatial_index)
							prev[w][h][d] = prev_ind
							state[w][h][d] = 1

						elif (state[w][h][d] == 1):
							if(phi[w][h][d] > new_dist):
								phi[w][h][d] = new_dist
								# spatial_index = spatial(w,h,d)
								result = find_adjust(trail_set, phi, new_dist, spatial_index)
								trail_set = result[0]
								trail_index[w][h][d] = result[1]
								prev[w][h][d] = prev_ind

	alive = np.array([])
	for w in range(size[0]):
		for h in range(size[1]):
			for d in range(size[2]):
				if state[w][h][d] == 2 and alive is not None:
					node = spatial(w,h,d)
					# node.set_parent(parent[w][h][d])
					alive = np.append(alive,node)
				elif state[w][h][d] == 2 and alive is None:
					node = spatial(w,h,d)
					# node.set_parent(parent[w][h][d])
					alive = np.asarray(node)
	print('alive: ',alive.size)



	ini_swc = []
	swc_map = np.empty((size[0],size[1],size[2]),dtype = np.int32)
	index = 1
	for i in alive:
		ini_swc.append([index,3,i.w,i.h,i.d,1,0])
		swc_map[i.w][i.h][i.d] = index
		i.index = index
		index+=1

	seed_loc = swc_map[seed_w][seed_h][seed_d]-1

	ini_swc[seed_loc][6] = -1

	for i in ini_swc:
		p_loc = parent[i[2]][i[3]][i[4]]
		# print(i[2],i[3],i[4])
		if i[6] == -1:
			continue
		else:
			i[6] = swc_map[p_loc.w][p_loc.h][p_loc.d]

	for i in alive:
		# print(i.parent)
		p = parent[i.w][i.h][i.d]
		if p == 0:
			i.set_parent(spatial(-1,-1,-1))
			i.parent.index = -1
			print('None parernt should be seed, ',i.w,i.h,i.d)
		else:
			i.set_parent(alive[swc_map[p.w][p.h][p.d]-1])
			i.parent.index = swc_map[i.parent.w][i.parent.h][i.parent.d]
		
	# print(ini_swc[0])
	ini_swc = np.asarray(ini_swc)
	swc_x = ini_swc[:, 2].copy()
	swc_y = ini_swc[:, 3].copy()
	ini_swc[:, 2] = swc_y
	ini_swc[:, 3] = swc_x

	saveswc(out_path+'new_fmtest_gap.swc', ini_swc)

	return alive




def test_heap():
	heap = FibonacciHeap()

	heap.insert([0.0,[2,3,4]])
	heap.insert([3.3,[5,4,6]])
	heap.insert([6.9,[344,3,2]])
	x = heap.extract_min()
	print(x.index)
	print(heap.total_nodes)