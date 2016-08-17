import argparse

from utils.io import *
from utils.heap import *
import skfmm

def main():
	parser = argparse.ArgumentParser(description='Arguments for see anisotropic filters.')
	# parser.add_argument('--file', type=str, default=None, required=True, help='The file to filter')
	parser.add_argument('--threshold', type=float, default=-1, help='threshold to distinguish the foreground and background; works on filtered image if --filter is enabled')
	parser.add_argument('--soma_threshold', type=float, default=-1, help='The threshold on the original image to get soma radius')
    
	# Argument for tracing

	parser.add_argument('--trace', dest='trace', action='store_true', help='Run tracing with APP2')
	parser.add_argument('--no-trace', dest='trace', action='store_false', help='Skip the tracing with APP2')
	parser.set_defaults(trace=True)

	# MISC
	parser.add_argument('--silence', dest='silence', action='store_true')
	parser.add_argument('--no-silence', dest='silence', action='store_false')
	parser.set_defaults(silence=False)

	args = parser.parse_args()


    # for testing using file name, need to change to args later
	img = loadimg('test/2.tif')
    
    # Distance Transform
	if args.trace:
		if args.soma_threshold < 0:
			try:
				from skimage import filters
			except ImportError:
				from skimage import filter as filters
			args.soma_threshold = filters.threshold_otsu(img)

		if not args.silence: 
			print('--DT to get soma location with threshold:', args.soma_threshold)
		
		# segment image
		bimg = (img > args.soma_threshold).astype('int')
		
		# boundary Distance Transform
		if not args.silence: 
			print('--Boundary DT...')
		dt = skfmm.distance(bimg,dx=1)
		
		dtmax = dt.max()
		print('DT max: ',dtmax)
		maxdpt = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
		marchmap = np.ones(img.shape)
		marchmap[maxdpt[0], maxdpt[1], maxdpt[2]] = -1


if __name__ == "__main__":
    main()