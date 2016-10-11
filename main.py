import argparse
from scipy import ndimage	

import numpy as np
from utils.io import *
from fm import *
from fm_new import *
import skfmm
np.set_printoptions(threshold=np.inf)

def main():
	parser = argparse.ArgumentParser(description='Arguments for see anisotropic filters.')
	# parser.add_argument('--file', type=str, default=None, required=True, help='The file to filter')
	parser.add_argument('--threshold', type=float, default=-1, help='threshold to distinguish the foreground and background; works on filtered image if --filter is enabled')
	parser.add_argument('--soma_threshold', type=float, default=-1, help='The threshold on the original image to get soma radius')
    
	# Argument for tracing

	parser.add_argument('--trace', dest='trace', action='store_true', help='Run tracing with APP2')
	parser.add_argument('--no-trace', dest='trace', action='store_false', help='Skip the tracing with APP2')
	parser.set_defaults(trace=True)

	parser.add_argument('--dt', dest='trace', action='store_true', help='Perform dt when fast marching')
	parser.add_argument('--no-dt', dest='trace', action='store_false', help='Skip dt')
	parser.set_defaults(dt=True)

	# MISC
	parser.add_argument('--silence', dest='silence', action='store_true')
	parser.add_argument('--no-silence', dest='silence', action='store_false')
	parser.set_defaults(silence=False)

	args = parser.parse_args()


    # for testing using file name, need to change to args later
	img = loadimg('test/crop2/crop2.tif')
	# print(img)
	size = img.shape
	print('--input image size: ', size)

    # Distance Transform
	if args.trace:
		# get soma location
		# if args.soma_threshold < 0:
		# 	try:
		# 		from skimage import filters
		# 	except ImportError:
		# 		from skimage import filter as filters
		# 	args.soma_threshold = filters.threshold_otsu(img)

		# if not args.silence: 
		# 	print('--DT to get soma location with threshold: ', args.soma_threshold)
		
		# print(np.where(img > 0))
		# segment image
		print('--original img')
		# print(img,img.shape)
		print('--segment img')
		# print(bimg,bimg.shape)

		print('--DT')
		# average_intensity = np.sum(img) / (size[0] * size[1] * size[2])
		# print('--Max intensity',max_intensity)
		# print('--average intensity: ',average_intensity)
		bimg = (img > args.threshold).astype('int')
		dt_result = skfmm.distance(bimg, dx=1)
		# dt_result = ndimage.distance_transform_edt(bimg,sampling=[1,1,1])

		print('--dt result size: ', dt_result.shape)

		# print(bimg[np.where(bimg == 1)])
		# print('--bimg')
		# print(bimg[36][30][6])
		count = 0
		print('--DT result')
		# print(dt_result)

		max_dt1 = np.max(dt_result)
		max_w = 0
		max_h = 0
		max_d = 0
		max_dt = 0
		for w in range(size[0]):
			for h in range(size[1]):
				for d in range(size[2]):
					if dt_result[w][h][d] > max_dt:
						max_dt = dt_result[w][h][d]
						max_w = w
						max_h = h
						max_d = d

		print('--source_index', max_dt,max_dt1,max_w,max_h,max_d)
		
		################    test number of neighbours      #####################
		# neighbours = get_neighbours(vertices,size[0]-1,size[1]-1,size[2]-1,size)
		# count = 0
		# for i in neighbours:
		# 	if i is not None:
		# 		count+=1
		# print('--number of neighbours: ')
		# print(count)
		
		# print('--Find trial set')
		# trials = find_trial_set(vertices,max_w,max_h,max_d,size)
		# trial_set = np.array(trials)

		# print(len(trial_set))
		# for i in trial_set:
		# 	print(i.w, i.h,i.d,i.dt,i.state)

		print('--Initial reconstruction')
		fastmarching_dt_tree(img,bimg,size,max_w,max_h,max_d)
		# fastmarching(img,bimg,dt_result,size)




		if not args.silence:
			print('--Hierarchy Prune')



def makespeed(dt, threshold=0):
    F = dt ** 4
    F[F<=threshold] = 1e-10
    return F

if __name__ == "__main__":
    main()