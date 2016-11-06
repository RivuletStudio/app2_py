from utils.io import *
import argparse
import sys

def main():
	input_file = sys.argv[1]
	input = loadswc(input_file)
	input[:, 5] =  1
	print(input[0])
	count = 0
	for i in input[6]:
		if i == 12601:
			count += 1
	print('neighbors amount of seed: ',count)
	saveswc(input_file,input)


if __name__ == "__main__":
    main()