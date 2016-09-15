import argparse
from scipy import ndimage   

import numpy as np
from utils.io import *

def main():
    swc = loadswc('test/1resmapled_2.tif_ini.swc')
    print(swc)
    print(1)