import argparse
from scipy import ndimage	

import numpy as np
from utils.io import *
from fast_marching import *
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
	img = loadimg('test/1resampled.tif')
	print('--input image size: ', img.shape)
	# print(img)
	size = img.shape
	average_intensity = np.sum(img) / (size[0] * size[1] * size[2])
	print('--average internsity: ',average_intensity)
	dtimg = (img > average_intensity).astype('int')

    # Distance Transform
	if args.trace:
		# get soma location
		if args.soma_threshold < 0:
			try:
				from skimage import filters
			except ImportError:
				from skimage import filter as filters
			args.soma_threshold = filters.threshold_otsu(img)

		if not args.silence: 
			print('--DT to get soma location with threshold: ', args.soma_threshold)
		
		# print(np.where(img > 0))
		# segment image
		bimg = (img > args.soma_threshold).astype('int')
		print('--original img')
		# print(img,img.shape)
		print('--segment img')
		# print(bimg,bimg.shape)

		print('--DT')
		result_dt = ndimage.morphology.distance_transform_edt(dtimg,sampling=[1,1,1])
		# print([np.where(bimg > 0)])
		print('--bimg')
		print(dtimg[30])
		print('--DT result')
		print(result_dt[30])

		vertices = []
		index = 0
		print('--FM')
		for i in range (size[0]):
			for j in range (size[1]):
				for k in range (size[2]):
					flag = 'FAR'
					if bimg[i][j][k] == 1:
						flag = 'ALIVE'
					element = vertex(index, i, j, k, dtimg[i][j][k], img[i][j][k], flag)
					vertices.append(element)

		print('--Vertex dt')
		for i in vertices:
			if i.dt != 0 and i.dt != 1:
				print(i.dt)

		# for i in vertices:
			# if i.state != 'FAR':
			# if i.dt != 1 and i.dt != 0:
				# print(i.ind, i.x, i.y, i.z, i.dt, i.intensity, i.state)
		

		# # boundary Distance Transform
		# if not args.silence: 
		# 	print('--Boundary DT...')
		# dt = skfmm.distance(bimg,dx=1)
		# print(dt.shape)
		# print(dt)



		# dtmax = dt.max()
		# print('DT max: ',dtmax)
		# maxdpt = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
		# marchmap = np.ones(img.shape)
		# marchmap[maxdpt[0], maxdpt[1], maxdpt[2]] = -1

		# # Fast Marching
		# if not args.silence:
		# 	print('--Fast Marching...')
		# t = skfmm.travel_time(marchmap, makespeed(dt), dx=5e-3)
		# # print(t)

		if not args.silence:
			print('--Hierarchy Prune')



def makespeed(dt, threshold=0):
    F = dt ** 4
    F[F<=threshold] = 1e-10
    return F

if __name__ == "__main__":
    main()