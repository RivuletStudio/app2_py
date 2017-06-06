from anfilter import *
from io import * 
import matplotlib.pyplot as plt
from scipy import io as sio
import os

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

	ostu_img = 0.
	print(img.shape)

	radii = np.arange(1,1.5,0.2)
	rho = 0.5

	# oof_matlab = mat['oof']
	# ostu_matlaboof = filters.threshold_otsu(oof_matlab)

	rps,_,_ = response(img.astype('float'), radii)
	thr = 1

	from scipy import ndimage as ndi
	from skimage import feature
	# canny = feature.canny(rps, sigma=3)
	rps *= (255/np.max(rps))
	rps = np.ceil(rps).astype(img.dtype)
	print(np.max(rps),np.min(rps),rps.shape)
	writetiff3d('rsp_2.tif',rps)

	smoothed_rps = gaussian_filter(rps, 0.5)
	# smoothed_rps[np.argwhere(smoothed_rps>12)] += 50
	writetiff3d('smoothed_rps_1.tif',smoothed_rps)
# ostu_smooth = filters.threshold_otsu(smoothed_rps)
	ostu_smooth = 1

	plotidx = 1
	plt.subplot(4, 4, plotidx)
	plt.imshow(rps.max(axis=0))
	plt.title('OOF Python MEM_SAVE YZ')
	plotidx += 1

	plt.subplot(4, 4, plotidx)
	plt.imshow(rps.max(axis=1))
	plt.title('OOF Python MEM_SAVE XZ')
	plotidx += 1
	
	plt.subplot(4, 4, plotidx)
	plt.imshow(rps.max(axis=2))
	plt.title('OOF Python MEM_SAVE XY')
	plotidx += 1

	plt.subplot(4, 4, plotidx)
	plt.imshow((rps > thr).max(axis=2))
	plt.title('OOF Python MEM_SAVE Otsu XY')
	plotidx += 1

	plt.subplot(4, 4, plotidx)
	plt.imshow(smoothed_rps.max(axis=0))
	plt.title('Smooth YZ')
	plotidx += 1

	plt.subplot(4, 4, plotidx)
	plt.imshow(smoothed_rps.max(axis=1))
	plt.title('Smooth XZ')
	plotidx += 1

	plt.subplot(4, 4, plotidx)
	plt.imshow(smoothed_rps.max(axis=2))
	plt.title('Smooth XY')
	plotidx += 1

	plt.subplot(4, 4, plotidx)
	plt.imshow((smoothed_rps > ostu_smooth).max(axis=2))
	plt.title('Smooth XY')
	plotidx +=1

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

	plt.subplot(4, 4, plotidx)
	plt.imshow(img.max(axis=0))
	plt.title('Original YZ')
	plotidx += 1

	plt.subplot(4, 4, plotidx)
	plt.imshow(img.max(axis=1))
	plt.title('Original XZ')
	plotidx += 1

	plt.subplot(4, 4, plotidx)
	plt.imshow(img.max(axis=2))
	plt.title('Original XY')
	plotidx += 1

	plt.subplot(4, 4, plotidx)
	plt.imshow((img > ostu_img).max(axis=2))
	plt.title('Original Otsu XY')
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

if __name__ == "__main__":
    main()
