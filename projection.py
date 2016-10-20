from utils.io import *
from libtiff import TIFFfile, TIFF
import scipy.misc
from PIL import Image, ImageDraw

def main():
	img = loadimg('test/1.tif')
	print(img)
	print('222')
	# x-y projection
	imgxy2d = img.max(axis=-1)
	# draw.line((100,200, 150,300), fill=128)

	# savetif('test/projection/projection_11.tif',img2d)
	scipy.misc.imsave('imgxy2d_projection.tif', imgxy2d)
	im = Image.open('imgxy2d.tif')
	draw = ImageDraw.Draw(im)
	draw.line((300,170, 360,170), fill = 128)
	draw.line((300,120, 360,120), fill = 128)
	im.show()
	writetiff3d('xxx.tif',imgxy2d)


if __name__ == "__main__":
    main()