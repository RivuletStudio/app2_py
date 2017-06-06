# from anfilter import *
from io import * 
import matplotlib.pyplot as plt
from scipy import io as sio
import os
import numpy as np
import skfmm

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from scipy.ndimage.filters import gaussian_filter

# mat = sio.loadmat('tests/data/very-small-oof.mat', )
# img = mat['img']
def main():
	from libtiff import TIFFfile, TIFF
	tiff = TIFF.open('2.tif', mode='r')
	stack = []
	for sample in tiff.iter_images():
	    stack.append(np.flipud(sample))

	out = np.dstack(stack)
	tiff.close()
	img=out

	bimg = (img > 22).astype('int')
		# print('--Finished: %.2f sec.' % (time.time() - starttime))
	dt_result = skfmm.distance(bimg, dx=5e-2)
		# print('--Finished: %.2f sec.' % (time.time() - starttime))

		# find seed location (maximum intensity node)
	max_dt = np.max(dt_result)
	seed_location = np.argwhere(dt_result==np.max(dt_result))[0]
	max_intensity = img[seed_location[0]][seed_location[1]][seed_location[2]]
	# print('--seed index',max_dt,max_intensity,seed_location[0],seed_location[1],seed_location[2])

		# dt_result = skfmm.distance(np.logical_not(dt_result), dx=5e-3)
	dt_result[dt_result > 0.04] = 0.04
		# dt_result = max_dt-dt_result
	speed = makespeed(dt_result)
	marchmap = np.ones(bimg.shape)
	marchmap[seed_location[0]][seed_location[1]][seed_location[2]] = -1
	timemap = skfmm.travel_time(marchmap, speed, dx=5e-3)
	print(timemap.shape,type(timemap))
	# print(timemap[0])

	fm_result = loadswc('1.swc')

	swc_x = fm_result[:, 2].copy()
	swc_y = fm_result[:, 3].copy()
	fm_result[:, 2] = swc_y
	fm_result[:, 3] = swc_x

	# for i in fm_result:
	# 	if img[int(i[])][int(i[3])][int(i[4])] >= 22:
	# 		timemap[int(i[2])][int(i[3])][int(i[4])] = 1e+10

	for i in timemap:
		i = 3.8e+10

	far = np.argwhere(bimg == 1)
	back = np.argwhere(bimg == 0)
	print(far[0])
	# print(fm_position[0])

	for i in back:
		timemap[int(i[0])][int(i[1])][int(i[2])] = 1e-10

	for i in far:
		timemap[int(i[0])][int(i[1])][int(i[2])] = 9e+10

	for i in fm_result:
		timemap[int(i[2])][int(i[3])][int(i[4])] = 1e+10


	fig, ax = plt.subplots()
	tm = ax.imshow(timemap.max(axis=2))
	plt.colorbar(tm)

	plt.show()

def makespeed(dt):
    '''
    Make speed image for FM from distance transform
    '''

    F = dt**4
    F[F <= 0] = 1e-10

    return F

def loadswc(filepath):
    swc = []
    with open(filepath) as f:
        lines = f.read().split('\n')
        for l in lines:
            if not l.startswith('#'):
                cells = l.split(' ')
                if len(cells) ==7:
                    cells = [float(c) for c in cells]
                    cells[2:5] = [c-1 for c in cells[2:5]]
                    swc.append(cells)
    return np.array(swc)

if __name__ == "__main__":
    main()

