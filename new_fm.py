import numpy as np
import numpy.linalg
import math
import time
from utils.io import *
from node import *

givals = [
    22026.5, 20368, 18840.3, 17432.5, 16134.8, 14938.4, 13834.9, 12816.8,
    11877.4, 11010.2, 10209.4, 9469.8, 8786.47, 8154.96, 7571.17, 7031.33,
    6531.99, 6069.98, 5642.39, 5246.52, 4879.94, 4540.36, 4225.71, 3934.08,
    3663.7, 3412.95, 3180.34, 2964.5, 2764.16, 2578.14, 2405.39, 2244.9,
    2095.77, 1957.14, 1828.24, 1708.36, 1596.83, 1493.05, 1396.43, 1306.47,
    1222.68, 1144.62, 1071.87, 1004.06, 940.819, 881.837, 826.806, 775.448,
    727.504, 682.734, 640.916, 601.845, 565.329, 531.193, 499.271, 469.412,
    441.474, 415.327, 390.848, 367.926, 346.454, 326.336, 307.481, 289.804,
    273.227, 257.678, 243.089, 229.396, 216.541, 204.469, 193.129, 182.475,
    172.461, 163.047, 154.195, 145.868, 138.033, 130.659, 123.717, 117.179,
    111.022, 105.22, 99.7524, 94.5979, 89.7372, 85.1526, 80.827, 76.7447,
    72.891, 69.2522, 65.8152, 62.5681, 59.4994, 56.5987, 53.856, 51.2619,
    48.8078, 46.4854, 44.2872, 42.2059, 40.2348, 38.3676, 36.5982, 34.9212,
    33.3313, 31.8236, 30.3934, 29.0364, 27.7485, 26.526, 25.365, 24.2624,
    23.2148, 22.2193, 21.273, 20.3733, 19.5176, 18.7037, 17.9292, 17.192,
    16.4902, 15.822, 15.1855, 14.579, 14.0011, 13.4503, 12.9251, 12.4242,
    11.9464, 11.4905, 11.0554, 10.6401, 10.2435, 9.86473, 9.50289, 9.15713,
    8.82667, 8.51075, 8.20867, 7.91974, 7.64333, 7.37884, 7.12569, 6.88334,
    6.65128, 6.42902, 6.2161, 6.01209, 5.81655, 5.62911, 5.44938, 5.27701,
    5.11167, 4.95303, 4.80079, 4.65467, 4.51437, 4.37966, 4.25027, 4.12597,
    4.00654, 3.89176, 3.78144, 3.67537, 3.57337, 3.47528, 3.38092, 3.29013,
    3.20276, 3.11868, 3.03773, 2.9598, 2.88475, 2.81247, 2.74285, 2.67577,
    2.61113, 2.54884, 2.48881, 2.43093, 2.37513, 2.32132, 2.26944, 2.21939,
    2.17111, 2.12454, 2.07961, 2.03625, 1.99441, 1.95403, 1.91506, 1.87744,
    1.84113, 1.80608, 1.77223, 1.73956, 1.70802, 1.67756, 1.64815, 1.61976,
    1.59234, 1.56587, 1.54032, 1.51564, 1.49182, 1.46883, 1.44664, 1.42522,
    1.40455, 1.3846, 1.36536, 1.3468, 1.3289, 1.31164, 1.29501, 1.27898,
    1.26353, 1.24866, 1.23434, 1.22056, 1.2073, 1.19456, 1.18231, 1.17055,
    1.15927, 1.14844, 1.13807, 1.12814, 1.11864, 1.10956, 1.10089, 1.09262,
    1.08475, 1.07727, 1.07017, 1.06345, 1.05709, 1.05109, 1.04545, 1.04015,
    1.03521, 1.0306, 1.02633, 1.02239, 1.01878, 1.0155, 1.01253, 1.00989,
    1.00756, 1.00555, 1.00385, 1.00246, 1.00139, 1.00062, 1.00015, 1
]

def GI(w,h,d, img, max_intensity, min_intensity):
    return givals[(int)((img[w][h][d] - min_intensity) /
                        max_intensity * 255)]

"""
initial tree reconsturction using fast-marching

"""
def fastmarching(img, bimg, size, seed_w, seed_h, seed_d, max_intensity,threshold,
                         allow_gap, out_path):

    # starttime = time.time()
    # state 0 for FAR, state 1 for TRAIL, state 2 for ALIVE
    state = np.zeros((size[0], size[1], size[2]))

    # initialize 
    phi = np.empty((size[0], size[1], size[2]), dtype=np.float32)
    parent = np.empty((size[0], size[1], size[2]), dtype=np.int32)
    prev = np.empty((size[0], size[1], size[2]), dtype=np.int32)
    swc_index = np.empty((size[0], size[1], size[2]), dtype=np.int32)


    # starttime = time.time()
    for i in range(size[0]):
        phi[i,:,:] = np.inf
    # print('--Cost2: %.2f sec.' % (time.time() - starttime))                
    # print('finish assigning values')

    current_index = 0
    # put seed into ALIVE set
    state[seed_w][seed_h][seed_d] = 2
    phi[seed_w][seed_h][seed_d] = 0.0
    swc_index[seed_w][seed_h][seed_d] = 1
    prev[seed_w][seed_h][seed_d] = 1

    # [phi,w,h,d,par_id]
    trail_set = np.asarray([[0,seed_w, seed_h, seed_d]])
    print(trail_set)
    # [id,radius,w,h,d,1,par_id]
    alive_set = np.asarray([[]])
    # alive_set = np.asarray([1,1,seed_w, seed_h, seed_d,1,-1])
    # print(trail_set.size)
    # print(alive_set.size)
    starttime = time.time()
    totaltime = 0
    counter = 0
    while (trail_set.size != 0):
        counter+=1
        # print('size: ',trail_set.shape)
        # if(trail_set.shape[0] == 1):
        #     min_ind = trail_set 
        # else:
        min_ind = trail_set[0,:]

        # print('min_ind')
        # print(min_ind)
        trail_set = numpy.delete(trail_set, (0), axis=0)
        i = int(min_ind[1])
        j = int(min_ind[2])
        k = int(min_ind[3])
        prev_ind = prev[i][j][k]
        # if(trail_set.shape[0] == 1):
        #     min_ind = trail_set 
        # else:
        parent[i][j][k] = prev_ind

        state[i][j][k] = 2
        swc_index[i][j][k] = current_index
        insert_swc = time.time()
        if(alive_set.shape[1] == 0):
            alive_set = np.asarray([[current_index,3,i,j,k,1,0]])
        else:
            alive_set = np.vstack((alive_set,[current_index,3,i,j,k,1,prev_ind]))
        current_index += 1
        # print('insert takes %.10f sec.' % (time.time()-insert_swc))
        totaltime+=(time.time()-insert_swc)

        for kk in range(-1, 2):
            d = k + kk
            # if (d < 0 or d >= size[2]):
            #     continue
            for jj in range(-1, 2):
                h = j + jj
                # if (h < 0 or h >= size[1]):
                #     continue
                for ii in range(-1, 2):
                    w = i + ii
                    # if (w < 0 or w >= size[0]):
                    #     continue

                    offset = abs(ii) + abs(jj) + abs(kk)
                    # print('offset: ',offset)
                    # this 2 is cnn type
                    if offset == 0 or offset > 2:
                        continue

                    factor = 1
                    if offset == 2:
                        factor = 1.414214

                    if (img[w][h][d] <= threshold and
                            img[i][j][k] <= threshold):
                        continue

                    # spatial_index = spatial(w, h, d)
                    if (state[w][h][d] != 2):
                        # min_intensity set as 0
                        new_dist = phi[w][h][d] + (GI(
                            w,h,d, img, max_intensity, 0.0) + GI(
                                i,j,k, img, max_intensity, 0.0)
                                                   ) * factor * 0.5
                        prev_ind = swc_index[i][j][k]

                        if (state[w][h][d] == 0):
                            phi[w][h][d] = new_dist
                            # insert into trail set
                            if trail_set.shape[1] == 0:
                                trail_set = np.vstack((trail_set,[new_dist,w,h,d]))
                            else:
                                sort_time = time.time()
                                trail_set = np.vstack((trail_set,[new_dist,w,h,d]))
                                trail_set[np.argsort(trail_set[:,0])]
                                # print('insert takes: %.2f',time.time()-sort_time)

                            prev[w][h][d] = prev_ind
                            state[w][h][d] = 1

                        elif (state[w][h][d] == 1):
                            if (phi[w][h][d] > new_dist):
                                phi[w][h][d] = new_dist
                                # spatial_index = spatial(w,h,d)
                                temp_ind = np.argwhere((trail_set[:,1] == w) & (trail_set[:,2] == h) & (trail_set[:,3] == d))[0]
                                trail_set[temp_ind][0] = new_dist
                                trail_set[np.argsort(trail_set[:,0])]
                                sort_time = time.time()
                                # print('adjust takes: %.2f',time.time()-sort_time)
                                prev[w][h][d] = prev_ind

    print(totaltime)
    print(counter)
    print('--FM finished')
    print('--Fast Marching: %.2f sec.' % (time.time() - starttime))

    starttime = time.time()
    swc_x = alive_set[:, 2].copy()
    swc_y = alive_set[:, 3].copy()
    alive_set[:, 2] = swc_y
    alive_set[:, 3] = swc_x
    saveswc(out_path + 'new_fm_ini.swc',alive_set)
    # print('--Start: %.2f sec.' % (starttime))
    print('--Store ini_swc: %.2f sec.' % (time.time() - starttime))
    # print('--FM finished')

    return alive_set