from utils.io import *
import argparse
import sys

def main():
	input_file = sys.argv[1]
	input = loadswc(input_file)
	input[:, 5] =  1
	print(input[0])
	saveswc(input_file,input)


if __name__ == "__main__":
    main()