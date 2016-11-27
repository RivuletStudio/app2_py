import argparse
from scipy import ndimage	

import numpy as np
from utils.io import *
from new_fm import *
from new_hp import *
# from hierarchy_prune import *
import skfmm
np.set_printoptions(threshold=np.inf)

def main():
	parser = argparse.ArgumentParser(description='Arguments for see anisotropic filters.')
	parser.add_argument('--file', type=str, default=None, required=True, help='The path of input file')
	parser.add_argument('--out', type=str, default=None, required=True, help='The out path of output swc')
	parser.add_argument('--threshold', type=float, default=-1, help='threshold to distinguish the foreground and background; works on filtered image if --filter is enabled')
	parser.add_argument('--soma_threshold', type=float, default=-1, help='The threshold on the original image to get soma radius')
    
	# Argument for tracing

	parser.add_argument('--allow_gap', dest='allow_gap', action='store_true', help='allow gap during tracing')
	parser.add_argument('--no-allow_gap', dest='allow_gap', action='store_false', help='allow no gap during tracing')
	parser.set_defaults(allow_gap=True)

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
	img = loadimg(args.file)
	# print(img)
	size = img.shape
	print('--input image size: ', size)

    # Distance Transform
	if args.trace:
		print('--DT')
		# average_intensity = np.sum(img) / (size[0] * size[1] * size[2])
		# print('--Max intensity',max_intensity)
		# print('--average intensity: ',average_intensity)
		bimg = (img > args.threshold).astype('int')
		dt_result = skfmm.distance(bimg, dx=1)
		# dt_result = ndimage.distance_transform_edt(bimg,sampling=[1,1,1])

		print('--dt result size: ', dt_result.shape)

		count = 0
		print('--DT result')

		# find seed location
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

		print('--seed_index', max_dt,max_dt1,max_w,max_h,max_d)

		print('--Initial reconstruction by Fast Marching')

		# test_heap()
		alive = fastmarching(img,bimg,size,max_w,max_h,max_d,args.threshold,args.allow_gap,args.out)


		# new_hp(img,alive,args.out)
		hp(img,size,alive,args.out,args.threshold)

		

		# if not args.silence:
			# print('--Hierarchy Prune')
			# new_hp(ini_swc,img,size)
			# hierarchy_prune(ini_swc,img,size,args.out,args.threshold)

if __name__ == "__main__":
    main()