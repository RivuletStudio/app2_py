# from anfilter import *
from io import * 
import matplotlib.pyplot as plt
from scipy import io as sio
import os
import skfmm
import numpy as np

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

	# plotidx = 1
	# plt.subplot(4, 4, plotidx)
	# plt.imshow(timemap.max(axis=0))
	# plt.title('1')
	# plotidx += 1

	# plotidx = 1
	# plt.subplot(4, 4, plotidx)
	# plt.imshow(timemap.max(axis=1))
	# plt.title('2')
	# plotidx += 1

	# plotidx = 1
	# plt.subplot(4, 4, plotidx)
	# plt.imshow(timemap.max(axis=2))
	# plt.title('3')
	# plotidx += 1

	# plotidx = 1
	# plt.subplot(4, 4, plotidx)
	# plt.imshow(timemap.max(axis=2))
	# plt.title('4')
	# plotidx += 1
	# plt.colorbar(timemap)

	# fig, ax = plt.subplots()
	# tm = ax.imshow(timemap.max(axis=2))
	# plt.colorbar(tm)

	fig, ax = plt.subplots()
	tm = ax.imshow(timemap.max(axis=2))
	plt.colorbar(tm)

	plt.show()

# ostu_smooth = filters.threshold_otsu(smoothed_rps)
	# ostu_smooth = 1

	# plotidx = 1
	# plt.subplot(4, 4, plotidx)
	# plt.imshow(rps.max(axis=0))
	# plt.title('OOF Python MEM_SAVE YZ')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow(rps.max(axis=1))
	# plt.title('OOF Python MEM_SAVE XZ')
	# plotidx += 1
	
	# plt.subplot(4, 4, plotidx)
	# plt.imshow(rps.max(axis=2))
	# plt.title('OOF Python MEM_SAVE XY')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow((rps > thr).max(axis=2))
	# plt.title('OOF Python MEM_SAVE Otsu XY')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow(smoothed_rps.max(axis=0))
	# plt.title('Smooth YZ')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow(smoothed_rps.max(axis=1))
	# plt.title('Smooth XZ')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow(smoothed_rps.max(axis=2))
	# plt.title('Smooth XY')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow((smoothed_rps > ostu_smooth).max(axis=2))
	# plt.title('Smooth XY')
	# plotidx +=1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow(canny.max(axis=0))
	# plt.title('OOF Matlab YZ')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow(canny.max(axis=1))
	# plt.title('OOF Matlab XZ')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow(canny.max(axis=2))
	# plt.title('OOF Matlab XY')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow((canny > ostu_matlaboof).max(axis=2))
	# plt.title('OOF Matlab Otsu XY')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow(img.max(axis=0))
	# plt.title('Original YZ')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow(img.max(axis=1))
	# plt.title('Original XZ')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow(img.max(axis=2))
	# plt.title('Original XY')
	# plotidx += 1

	# plt.subplot(4, 4, plotidx)
	# plt.imshow((img > ostu_img).max(axis=2))
	# plt.title('Original Otsu XY')
	plt.show()

def loadimg(file):
    if file.endswith('.mat'):
        filecont = sio.loadmat(file)
        img = filecont['img']
        for z in range(img.shape[-1]): # Flip the image upside down
            img[:,:,z] = np.flipud(img[:,:,z])
        img = np.swapaxes(img, 0, 1)
    elif file.endswith('.tif'):
        img = loadtiff3d(file)
    elif file.endswith('.nii') or file.endswith('.nii.gz'):
        import nibabel as nib
        img = nib.load(file)
        img = img.get_data()

def loadtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""
    from libtiff import TIFFfile, TIFF
    tiff = TIFF.open(filepath, mode='r')
    stack = []
    for sample in tiff.iter_images():
        stack.append(np.flipud(sample))

    out = np.dstack(stack)
    tiff.close()
    print(out.shape)

    return out

def writetiff3d(filepath, block):
    from libtiff import TIFFfile, TIFF
    try:
        os.remove(filepath)
    except OSError:
        pass

    tiff = TIFF.open(filepath, mode='w')
    block = np.flipud(block)
    
    for z in range(block.shape[2]):
        tiff.write_image(block[:,:,z], compression=None)
    tiff.close()

def makespeed(dt):
    '''
    Make speed image for FM from distance transform
    '''

    F = dt**4
    F[F <= 0] = 1e-10

    return F

if __name__ == "__main__":
    main()
