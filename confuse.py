from utils.metrics import *
from utils.io import *
import argparse

def main():
	parser = argparse.ArgumentParser(description='Arguments for app2_py.')
	parser.add_argument('-f', type=str, default=None, required=True, help='The path of compared swc')
	parser.add_argument('-g', type=str, default=None, required=True, help='The path of ground truth')
	args = parser.parse_args()
	result = loadswc(args.f)
	ground_truth = loadswc(args.g)
	precision_recall(result,ground_truth,4,4)


if __name__ == "__main__":
    main()