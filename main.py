import argparse
from scipy import ndimage	
import time
import numpy as np
from utils.io import *
from new_fm import *
from new_hp import *
# from hierarchy_prune import *
import skfmm
np.set_printoptions(threshold=np.inf)

def main():
	parser = argparse.ArgumentParser(description='Arguments for app2_py.')
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

	starttime = time.time()
	print('--load image')
	img = loadimg(args.file)

	print('--crop image')
	# img = crop(img,args.threshold)[0]
	print('--save crop image')
	# writetiff3d(args.out+'crop.tif',img)
	size = img.shape
	print('--input image size: ', size)

    # Distance Transform
	if args.trace:
		print('--DT to find soma location')
		bimg = (img > args.threshold).astype('int')
		# print('--Finished: %.2f sec.' % (time.time() - starttime))
		dt_result = skfmm.distance(bimg, dx=5e-2)
		# print('--Finished: %.2f sec.' % (time.time() - starttime))

		# find seed location (maximum intensity node)
		max_dt = np.max(dt_result)
		seed_location = np.argwhere(dt_result==np.max(dt_result))[0]
		max_intensity = img[seed_location[0]][seed_location[1]][seed_location[2]]
		print('--seed index',max_dt,max_intensity,seed_location[0],seed_location[1],seed_location[2])

		# marchmap = np.ones(size)
		# marchmap[seed_location[0]][seed_location[1]][seed_location[2]] = -1
		# t = skfmm.travel_time(marchmap,makespeed(dt_result),dx=5e-3)

		print('--SKFMM: %.2f sec.' % (time.time() - starttime))
		print('--initial reconstruction by Fast Marching')
		alive = fastmarching(img,bimg,size,seed_location[0],seed_location[1],seed_location[2],max_intensity,args.threshold,args.allow_gap,args.out)
		print('--initial reconstruction finished')
		print('--FM Total: %.2f sec.' % (time.time() - starttime))
		# alive = np.array([])

		starttime2 = time.time()
		print('--perform hierarchical pruning')
		hp(img,bimg,size,alive,args.out,args.threshold)
		print('--APP2 finished')
		print('--Pruning: %.2f sec.' % (time.time() - starttime2))
		print('--Finished: %.2f sec.' % (time.time() - starttime))

		

def makespeed(dt, threshold=0):
    '''
    Make speed image for FM from distance transform
    '''

    F = dt**4
    F[F <= threshold] = 1e-10

    return F

def crop(img, thr):
    """Crop a 3D block with value > thr"""
    ind = np.argwhere(img > thr)
    x = ind[:, 0]
    y = ind[:, 1]
    z = ind[:, 2]
    xmin = max(x.min() - 10, 0)
    xmax = min(x.max() + 10, img.shape[0])
    ymin = max(y.min() - 10, 1)
    ymax = min(y.max() + 10, img.shape[1])
    zmin = max(z.min() - 10, 2)
    zmax = min(z.max() + 10, img.shape[2])

    return img[xmin:xmax, ymin:ymax, zmin:zmax], np.array(
        [[xmin, xmax], [ymin, ymax], [zmin, zmax]])


if __name__ == "__main__":
    main()