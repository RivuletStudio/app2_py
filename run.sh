#!/bin/sh 
# python3 main.py --file test/zebra/2.tif --threshold 24 --out test/zebra/no_smooth_result_2.swc
python3 main.py --file test/2000-1/1.tif --threshold 0 --out test/2000-1/no_smooth_result.swc
# python3 confuse.py -f test/2000-4/new_result_3_4.swc -g test/2000-4/rivulet_result.swc