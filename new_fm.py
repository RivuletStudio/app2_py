import numpy as np
import numpy.linalg
import math
from utils.io import *
from node import *
from heap import *
from scipy.spatial import distance

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

def GI(index, img, max_intensity, min_intensity):
    return givals[(int)((img[index.w][index.h][index.d] - min_intensity) /
                        max_intensity * 255)]


#  find average and max intensity
def find_max_intensity(img, size):
    max_intensity = 0
    min_intensity = np.inf
    total_intensity = 0
    not_zero = 0
    max_w = 0
    max_h = 0
    max_d = 0
    for w in range(size[0]):
        for h in range(size[1]):
            for d in range(size[2]):
                if img[w][h][d] > max_intensity:
                    max_w = w
                    max_h = h
                    max_d = d
                    max_intensity = img[w][h][d]

                # print(img[w][h][d])
    print('total intensity: ', total_intensity)
    print('max intensity: ', max_intensity, max_w, max_d, max_h)
    print('total vertices: ', count)
    return max_intensity


def insert(trail_set, phi, new_dist, spatial):
    ind = 0
    if trail_set is None:
        trail_set = np.insert(trail_set, ind, spatial)
        # print('after insert: ',trail_set.size)
        return trail_set

    # print('size: ',trail_set.size)
    for i in trail_set:
        if new_dist < phi[i.w][i.h][i.d]:
            trail_set = np.insert(trail_set, ind, spatial)
            # print('after insert: ',trail_set.size)
            return trail_set
        ind += 1
    trail_set = np.insert(trail_set, ind, spatial)
    return trail_set


def find_adjust(trail_set, phi, new_dist, spatial):
    index = 0
    for i in trail_set:
        if (i.w == spatial.w and i.h == spatial.h and i.d == spatial.d):
            break
        index += 1

    trail_set = np.delete(trail_set, index)

    ind = 0

    for i in trail_set:
        if new_dist < phi[i.w][i.h][i.d]:
            trail_set = np.insert(trail_set, ind, spatial)
            return trail_set, ind
        ind += 1
    trail_set = np.insert(trail_set, ind, spatial)
    # print('after insert: ',trail_set.size)
    return


def app2(img, bimg, size, seed_w, seed_h, seed_d, threshold,
                         allow_gap, out_path):

    alive = fastmarching(img, bimg, size, seed_w, seed_h, seed_d, threshold,
                         allow_gap, out_path)
    hp(img,alive,out_path)



def fastmarching(img, bimg, size, seed_w, seed_h, seed_d, threshold,
                         allow_gap, out_path):
    max_intensity = np.amax(img)
    print('max intensity:', max_intensity)

    # state 0 for FAR, state 1 for TRAIL, state 2 for ALIVE
    state = np.zeros((size[0], size[1], size[2]))

    # initialize 
    phi = np.empty((size[0], size[1], size[2]), dtype=np.float32)
    parent = np.empty((size[0], size[1], size[2]), dtype=spatial)
    prev = np.empty((size[0], size[1], size[2]), dtype=spatial)
    # trail_set = np.array([])
    # trail_index = np.zeros((size[0],size[1],size[2]),dtype = np.int32)

    for w in range(size[0]):
        for h in range(size[1]):
            for d in range(size[2]):
                parent[w][h][d] = spatial(w, h, d)
                phi[w][h][d] = np.inf

    # put seed into ALIVE set
    state[seed_w][seed_h][seed_d] = 2
    phi[seed_w][seed_h][seed_d] = 0.0

    spatial_index = spatial(seed_w, seed_h, seed_d)
    trail_set = np.asarray([spatial_index])
    # print('11111size: ',trail_set.size)
    index = 0

    while (trail_set.size != 0):
        # print('size: ',trail_set.size)
        min_ind = trail_set.item(0)
        trail_set = np.delete(trail_set, 0)
        # print('size: ',trail_set.size)
        # print('after extract: ',trail_set.size)
        # min_ind = min_elem.index
        # print(min_ind)
        i = min_ind.w
        j = min_ind.h
        k = min_ind.d
        prev_ind = prev[i][j][k]

        parent[i][j][k] = prev_ind

        state[i][j][k] = 2

        for kk in range(-1, 2):
            d = k + kk
            if (d < 0 or d >= size[2]):
                continue
            for jj in range(-1, 2):
                h = j + jj
                if (h < 0 or h >= size[1]):
                    continue
                for ii in range(-1, 2):
                    w = i + ii
                    if (w < 0 or w >= size[0]):
                        continue

                    offset = abs(ii) + abs(jj) + abs(kk)
                    # print('offset: ',offset)
                    # this 2 is cnn type
                    if offset == 0 or offset > 2:
                        continue

                    factor = 1
                    if offset == 2:
                        factor = 1.414214
                    elif offset == 3:
                        factor = 1.732051

                    if (allow_gap):
                        if (img[w][h][d] <= threshold and
                                img[i][j][k] <= threshold):
                            continue
                    else:
                        if (img[w][h][d] <= threshold):
                            continue

                    spatial_index = spatial(w, h, d)
                    if (state[w][h][d] != 2):
                        # min_intensity set as 0
                        new_dist = phi[w][h][d] + (GI(
                            spatial_index, img, max_intensity, 0.0) + GI(
                                min_ind, img, max_intensity, 0.0)
                                                   ) * factor * 0.5
                        prev_ind = min_ind

                        if (state[w][h][d] == 0):
                            phi[w][h][d] = new_dist
                            # spatial_index = spatial(w,h,d)
                            trail_set = insert(trail_set, phi, new_dist,
                                               spatial_index)
                            prev[w][h][d] = prev_ind
                            state[w][h][d] = 1

                        elif (state[w][h][d] == 1):
                            if (phi[w][h][d] > new_dist):
                                phi[w][h][d] = new_dist
                                # spatial_index = spatial(w,h,d)
                                result = find_adjust(trail_set, phi, new_dist,
                                                     spatial_index)
                                trail_set = result[0]
                                trail_index[w][h][d] = result[1]
                                prev[w][h][d] = prev_ind

    print('--FM finished')

    print('--Store ini_swc')
    alive = np.asarray(spatial(seed_w, seed_h, seed_d))
    for w in range(size[0]):
        for h in range(size[1]):
            for d in range(size[2]):
                if state[w][h][d] == 2:
                    if (w != seed_w or h != seed_h or d != seed_d):
                        node = spatial(w, h, d)
                        # node.set_parent(parent[w][h][d])
                        alive = np.append(alive, node)
    print('alive: ', alive.size)

    ini_swc = []
    swc_map = np.empty((size[0], size[1], size[2]), dtype=np.int32)
    index = 0
    for i in alive:
        ini_swc.append([index + 1, 3, i.w, i.h, i.d, 1, 0])
        swc_map[i.w][i.h][i.d] = index
        i.index = index
        index += 1

    seed_loc = swc_map[seed_w][seed_h][seed_d]

    ini_swc[seed_loc][6] = -1

    for i in ini_swc:
        p_loc = parent[i[2]][i[3]][i[4]]
        # print(i[2],i[3],i[4])
        if i[6] == -1:
            continue
        else:
            i[6] = swc_map[p_loc.w][p_loc.h][p_loc.d]

    for i in alive:
        # print(i.parent)
        p = parent[i.w][i.h][i.d]
        if p is None:
            i.set_parent(None)
            print('None parent should be seed, ', i.w, i.h, i.d)
        else:
            i.set_parent(alive[swc_map[p.w][p.h][p.d]])
            i.parent.index = swc_map[i.parent.w][i.parent.h][i.parent.d]

    # print(ini_swc[0])
    ini_swc = np.asarray(ini_swc)
    saveswc(out_path + 'ini_norotate.swc', ini_swc)
    swc_x = ini_swc[:, 2].copy()
    swc_y = ini_swc[:, 3].copy()
    ini_swc[:, 2] = swc_y
    ini_swc[:, 3] = swc_x

    saveswc(out_path + 'new_fmtest_gap.swc', ini_swc)

    print('--FM finished')

    t = alive[100].parent
    print(swc_map[t.w][t.h][t.d])
    p = parent[alive[100].w][alive[100].h][alive[100].d]
    print(swc_map[p.w][p.h][p.d])

    return alive


def hp(img,bimg,size,alive,out,threshold):
    print('Hierarchical Prune...')

    filter_segs = swc2topo_segs(img,size,alive,out,threshold)
    # longest_segment(topo_dists, topo_leafs,alive,leaf_nodes)
    
    # dark nodes pruning
    # print('--dark node pruning')
    # dark_num_pruned = 1
    # iteration = 1
    # is_pruneable = np.zeros(filter_segs.size)

    # index = 0
    # while(dark_num_pruned > 0):
    #     dark_num_pruned = 0
    #     for seg in filter_segs:
    #         leaf_marker = seg.leaf
    #         root_marker = seg.root
    #         if leaf_marker == root_marker:
    #             continue
    #         if img[leaf_marker.w][leaf_marker.h][leaf_marker.d] <= threshold:
    #             seg.leaf_marker = leaf_marker.parent
    #             dark_num_pruned+=1
    #             is_pruneable[index] = 1
    #         else:
    #             is_pruneable[index] = 0
    #     index+=1

    print('--dark segments pruning')
    # dark segment pruning
    # delete_set = np.array([])
    # index = 0
    # for seg in filter_segs:
    #     leaf_marker = seg.leaf
    #     root_marker = seg.root
    #     if(leaf_marker == root_marker):
    #         delete_set =  np.append(delete_set,index)
    #     p = leaf_marker
    #     sum_int=tol_num=dark_num = 0.0
    #     while(1):
    #         intensity = img[leaf_marker.w][leaf_marker.h][leaf_marker.d]
    #         sum_int+=intensity
    #         tol_num+=1
    #         if (intensity <= threshold):
    #             dark_num+=1
    #         if (p == root_marker):
    #             break
    #         p = p.parent
    #     if (sum_int/tol_num <= threshold or dark_num/tol_num >= 0.2 and is_pruneable[index] == 0):
    #         delete_set =  np.append(delete_set,index)
    #     index+=1

    # print('delete size:', delete_set.size)
    # tmp_segs = np.array([])
    # index = 0
    # d_index = 0
    # for seg in filter_segs:
    #     # print(index,delete_set[0])
    #     if index == delete_set[0]:
    #         # print('delete')
    #         delete_set = np.delete(delete_set,0)
    #         continue
    #     else:
    #         tmp_segs = np.append(tmp_segs,seg)
    #     index+=1

    # filter_segs = tmp_segs
    # print('Current segments size: ',filter_segs.size)
    
    # calculate radius for every node
    print('--calculating radius for every node')
    index = 0
    for seg in filter_segs:
        # print(index)
        leaf_marker = seg.leaf
        root_marker = seg.root
        p = leaf_marker
        while(1):
            real_threshold = 40
            if (real_threshold < threshold):
                real_threshold = threshold
            p.radius = getradius(bimg,p.w,p.h,p.d)
            if (p == root_marker):
                break
            p = p.parent
        index+=1

    seg_swc = []

    index = 0
    for seg in filter_segs:
        seg_tree = seg.get_elements()
        for i in seg_tree:
            if i.parent is None:
                seg_swc.append([i.index, 3, i.w, i.h, i.d, i.radius, -1])
            else:
                seg_swc.append([i.index, 3, i.w, i.h, i.d, i.radius, i.parent.index])
            # seg = seg.parent
    seg_swc = np.asarray(seg_swc)
    swc_x = seg_swc[:, 2].copy()
    swc_y = seg_swc[:, 3].copy()
    seg_swc[:, 2] = swc_y
    seg_swc[:, 3] = swc_x
    saveswc(out+'length_threshold5_test.swc', seg_swc)


    print('--Hierarchical Pruning')
    result_segs = hierchical_prune(filter_segs,img,size)
    print('result size: ',result_segs.size)
    # sort_segs = []
    # for seg in filter_segs:
    #     sort_segs.append(seg)

    # # sort by the length of the segment
    # sort_segs.sort(key=lambda x:x.length, reverse=True)
    # sort_segs = np.asarray(sort_segs)
    # visited_segs = np.array([])
    # result_segs = np.array([])
    # tol_sum_sig = tol_sum_rdc = 0.0
    # sr_ratio = 1.0/3.0
    # tmpimg = img
    # print('tmp before not 0 num:',np.count_nonzero(tmpimg))
    # for seg in sort_segs:
    #     s_id = np.argwhere(visited_segs==seg.parent)
    #     # if seg.parent and s_id is not None :
    #     #     continue
    #     # if s_id.size != 0:
    #     #     print(s_id)
    #     #     continue
    #     # if (s_id.size != 0 and (s_id[0][1] == 0)):
    #     #     continue
    #     # if (seg.parent and visited_segs.size != 0 and visited_segs[-1] == seg.parent):
    #     #     continue
    #     leaf_marker = seg.leaf
    #     root_marker = seg.root

    #     sum_sig=sum_rdc=0

    #     p = leaf_marker
    #     while(1):
    #         if(tmpimg[p.w][p.h][p.d] == -1):
    #             sum_rdc += img[p.w][p.h][p.d]
    #             # print(img[p.w][p.h][p.d],sum_rdc)
    #         else:
    #         #     r = p.radius
    #         #     sum_sphere_size = 0.0
    #         #     sum_delete_size = 0.0
    #         #     for kk in range(-r,r+1,1):
    #         #         z1 = p.d + kk
    #         #         if z1 < 0 or z1 >= size[2]:
    #         #             continue
    #         #             for jj in range(-r,r+1,1):
    #         #                 y1 = p.h + jj
    #         #                 if y1 < 0 or y2 >= size[1]:
    #         #                     continue    
    #         #                     for ii in range(-r,r+1,1):
    #         #                         x1 = p.w + ii
    #         #                         if x1 < 0 or x2 >= size[0]:
    #         #                             continue
    #         #                         dst = ii*ii + jj*jj + kk*kk
    #         #                         if dst > rr:
    #         #                             continue
    #         #                         sum_sphere_size+=1
    #         #                         if (img[x1][y1][z1] != tmpimg[x1][y1][z1]):
    #         #                             sum_delete_size+=1
    #         #     # print('sum_delete_size: ',sum_sphere_size,'sum_sphere_size: ',sum_sphere_size)
    #         #     if(sum_sphere_size > 0 and sum_delete_size/sum_sphere_size > 0.1):
    #         #         sum_rdc += img[p.w][p.h][p.d]
    #         #     else:
    #         #         sum_sig += img[p.w][p.h][p.d]
    #             sum_sig += img[p.w][p.h][p.d]

    #         if p == root_marker:
    #             break
    #         p = p.parent

    #     # print(sum_rdc,sum_sig)
    #     if (sum_rdc == 0 or (sum_sig/sum_rdc >= sr_ratio and sum_sig >= size[0]*size[1]*size[2])):
    #         if sum_rdc != 0:
    #             print('lol',sum_rdc)

    #     if(seg.parent is None or sum_rdc == 0.0 or (sum_sig/sum_rdc >= sr_ratio and sum_sig >= size[0]*size[1]*size[2])):
    #         # print(tol_sum_rdc,tol_sum_sig)
    #         tol_sum_sig += sum_sig
    #         tol_sum_rdc += sum_rdc

    #         p = leaf_marker
    #         seg_markers = np.array([])
    #         while(1):
    #             if (tmpimg[p.w][p.h][p.d] != -1):
    #                 seg_markers = np.append(seg_markers,p)
    #             if p == root_marker:
    #                 break
    #             p = p.parent

    #         for marker in seg_markers:
    #             r = marker.radius
    #             if (r > 0):
    #                 rr = r*r
    #                 for kk in range(-r,r+1,1):
    #                     z1 = p.d + kk
    #                     if z1 < 0 or z1 >= size[2]:
    #                         continue
    #                     for jj in range(-r,r+1,1):
    #                         y1 = p.h + jj
    #                         if y1 < 0 or y1 >= size[1]:
    #                             continue    
    #                         for ii in range(-r,r+1,1):
    #                             x1 = p.w + ii
    #                             if x1 < 0 or x1 > size[0]:
    #                                 continue
    #                             dst = ii*ii + jj*jj + kk*kk
    #                             if dst > rr:
    #                                 continue
    #                             tmpimg[x1][y1][z1] = -1
    #         result_segs = np.append(result_segs,seg)
    #         visited_segs = np.append(visited_segs,seg)

    # print('after not 0 num:',np.count_nonzero(tmpimg))

    # print("result_seg size: ",result_segs.size)
    # print("filter_seg size: ",filter_segs.size)

    # print(tol_sum_rdc,tol_sum_sig)
    # print("R/S ratio: ", tol_sum_rdc/tol_sum_sig)
    # print('--Leaf node pruning')
    # current



    seg_swc = []

    count = 0
    index = 0

    for seg in result_segs:
        seg_tree = seg.get_elements()
        for i in seg_tree:
            # print(i.parent)
            if i.parent is None:
                seg_swc.append([i.index, 3, i.w, i.h, i.d, i.radius, -1])
                count += 1
            else:
                seg_swc.append([i.index, 3, i.w, i.h, i.d, i.radius, i.parent.index])
            # seg = seg.parent
            index+=1

    seg_swc = np.asarray(seg_swc)
    swc_x = seg_swc[:, 2].copy()
    swc_y = seg_swc[:, 3].copy()
    seg_swc[:, 2] = swc_y
    seg_swc[:, 3] = swc_x
    saveswc(out+'length_threshold5_test_result.swc', seg_swc)




    return


def swc2topo_segs(img,size,alive,out,threshold):
    tol_num = alive.size

    leaf_nodes = np.array([])
    child_no = np.zeros(tol_num, dtype=int)

    for i in alive:
        if i.parent is None:
            continue
        else:
            child_no[i.parent.index] += 1

    # calculate distance for every tree nodes
    index = 0
    for i in child_no:
        if i == 0:
            leaf_nodes = np.append(leaf_nodes, alive[index])

        index += 1

    print('leaf_size: ', leaf_nodes.size)

    print('child no of index 31: ', child_no[0])

    count = 0
    for i in alive:
        if i.parent is None:
            continue
        elif i.parent.index == 0:
            # print(i.w,i.h,i.d)
            count += 1
    print('child no of index 31: ', count)

    leaf_num = leaf_nodes.size




    # furthest leaf distance for each tree node
    topo_dists = np.zeros(tol_num)
    topo_leafs = np.empty(tol_num, dtype=spatial)

    # for leaf in leaf_nodes:
    #     child_node = leaf
    #     parent_node = child_node.parent
    #     cid = child_node.index
    #     # cid = np.argwhere(alive == child_node)
    #     topo_leafs[cid] = leaf
    #     topo_dists[cid] = img[leaf.w][leaf.h][leaf.d] / 255.0
    #     # topo_dists[cid] = 0

        
    for leaf in leaf_nodes:
        child_node = leaf
        parent_node = child_node.parent
        cid = child_node.index
        topo_leafs[cid] = leaf
        topo_dists[cid] = img[leaf.w][leaf.h][leaf.d] / 255.0
        while (parent_node):

            pid = parent_node.index
            # pid = np.argwhere(alive == parent_node)
            tmp_dst = img[parent_node.w][parent_node.h][
                parent_node.d] / 255.0 + topo_dists[cid]

            # tmp_dst = distance.euclidean(
            #     [parent_node.w, parent_node.h, parent_node.d],
            #     [child_node.w, child_node.h, child_node.d]) + topo_dists[cid]
            if (tmp_dst >= topo_dists[pid]):
                    # print(child_node.w,child_node.h,child_node.d,tmp_dst)
                topo_dists[pid] = tmp_dst
                topo_leafs[pid] = topo_leafs[cid]
            else:
                break

            child_node = parent_node
            cid = pid
            parent_node = parent_node.parent

    fp = np.argmax(topo_dists)
    fn = topo_leafs[fp]
    # fn2 = leaf_nodes[fp]
    print('furthest point location: ', fn.w, fn.h, fn.d, 'index: ',fp,'length: ',
          topo_dists[fp])
    print('seed topo to this point: ', topo_leafs[0].w, topo_leafs[0].h,
          topo_leafs[0].d, topo_dists[0])

    topo_segs = np.empty(leaf_num, dtype=segment)

    index = 0
    for leaf in leaf_nodes:
        root_marker = leaf
        root_parent = root_marker.parent
        level = 1

        while (root_parent and topo_leafs[root_parent.index] == leaf):
            # while (root_parent and topo_leafs[np.argwhere(alive == root_marker)] == leaf):
            if child_no[root_marker.index] >= 2:
                # if child_no[np.argwhere(alive == root_marker)] >= 2:
                level += 1
            root_marker = root_parent
            root_parent = root_marker.parent

        dst = topo_dists[root_marker.index]
        # dst = topo_dists[np.argwhere(alive == root_marker)]

        # if (leaf == root_marker and dst > 5):
        #     print('NONONONO!')
        #     print(dst)
        topo_seg = segment(leaf, root_marker, dst, level)
        topo_segs[index] = topo_seg

        if (root_parent is None):
            topo_seg.parent = None
        else:
            leaf_marker2 = topo_leafs[root_parent.index]
            # leaf_marker2 = topo_leafs[np.argwhere(alive == root_parent)]
            # print(np.argwhere(leaf_nodes ==
            #                                         leaf_marker2).shape)
            # print(type(topo_segs[np.argwhere(leaf_nodes ==
            #                                         leaf_marker2)]))
            loc = np.argwhere(leaf_nodes == leaf_marker2)
            # print(loc[0])
            topo_seg.parent = topo_segs[loc[0][0]]
            # print(type(topo_segs[loc[0][0]]))
        index += 1

    longest_segment(topo_dists, topo_leafs,alive,leaf_nodes,topo_segs)
    complete_segment(topo_dists, topo_leafs,alive,leaf_nodes,topo_segs)

    filter_segs = np.array([])
    print('Current Segments size:  ',topo_segs.size)
    print('--Prune by length threhold')

    for seg in topo_segs:
        # seg_length = np.append(seg_length,seg.length)
        if seg.length > 5:
            filter_segs = np.append(filter_segs, seg)

    print('Current Segments size:  ',filter_segs.size)
    # for i in topo_segs:
    #     if(i.parent is None):
            # print('lolxx')

    # for i in filter_segs:
    #     print(type(i.parent))
    return filter_segs


def hierchical_prune(filter_segs,img,size):
    sort_segs = []
    for seg in filter_segs:
        sort_segs.append(seg)

    tmpimg = img
    sort_segs.sort(key=lambda x:x.length, reverse=True)
    sort_segs = np.asarray(sort_segs)
    result_segs = np.array([])

    for seg in sort_segs:
        current = seg.leaf
        root = seg.root
        overlap = 0
        non_overlap = 0

        while (current != root):
            r = current.radius
            for kk in range(-r,r+1,1):
                z1 = current.d + kk
                if z1 < 0 or z1 >= size[2]:
                    continue
                for jj in range(-r,r+1,1):
                    y1 = current.h + jj
                    if y1 < 0 or y1 >= size[1]:
                        continue    
                    for ii in range(-r,r+1,1):
                        x1 = current.w + ii
                        if x1 < 0 or x1 > size[0]:
                            continue
                        if tmpimg[x1][y1][z1] == 0:
                            overlap+=1
                        else:
                            non_overlap+=1
            current = current.parent


        print(overlap/(overlap+non_overlap),overlap,non_overlap)
        if (overlap/(overlap+non_overlap) < 0.75):
            result_segs = np.append(result_segs,seg)

            current = seg.leaf
            root = seg.root
            overlap = 0
            non_overlap = 0


            while (current != root):
                r = current.radius
                for kk in range(-r,r+1,1):
                    z1 = current.d + kk
                    if z1 < 0 or z1 >= size[2]:
                        continue
                    for jj in range(-r,r+1,1):
                        y1 = current.h + jj
                        if y1 < 0 or y1 >= size[1]:
                            continue    
                        for ii in range(-r,r+1,1):
                            x1 = current.w + ii
                            if x1 < 0 or x1 > size[0]:
                                continue
                            # print('lol')
                            tmpimg[x1][y1][z1] = 0
                            print(tmpimg[x1][y1][z1])
                            
                current = current.parent

    print(np.sum(tmpimg == 0))
    return result_segs


def markerRadius(img,size,p,threshold):
    max_r = min(size[0]/2,size[1]/2,size[2]/2)

    for ir in range(1,int(max_r+1),1):
        total_num = background_num = 0
        
        for dz in range(-ir, ir+1,1):
            for dy in range(-ir, ir+1,1):
                for dx in range(-ir,ir+1,1):
                    total_num+=1
                    r = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if (r > ir-1 and r<=ir):
                        i = p.w+dx
                        if (i<0 or i>=size[0]):
                            return ir
                        j = p.h+dy
                        if (j<0 or j>=size[1]):
                            return ir
                        k = p.d+dz
                        if (k<0 or k>=size[2]):
                            return ir 

                        if (img[i][j][k] <= threshold):
                            background_num+=1
                            if (background_num/total_num > 0.001):
                                return ir
    return ir


def getradius(bimg, x, y, z):
    r = 0
    x = math.floor(x)
    y = math.floor(y)
    z = math.floor(z)

    while True:
        r += 1
        try:
            if bimg[max(x - r, 0):min(x + r + 1, bimg.shape[0]), max(y - r, 0):
                    min(y + r + 1, bimg.shape[1]), max(z - r, 0):min(
                        z + r + 1, bimg.shape[2])].sum() / (2 * r + 1)**3 < .6:
                break
        except IndexError:
            break

    return r


def complete_segment(topo_dists, topo_leafs,alive,leaf_nodes,topo_segs):
    l_swc = []
    sort_segs = []
    for seg in topo_segs:
        sort_segs.append(seg)

    # sort by the length of the segment
    sort_segs.sort(key=lambda x:x.length, reverse=True)
    sort_segs = np.asarray(sort_segs)
    # print('longest segs: ',sort_segs[0].length,'size: ',l_path.size)
    t = sort_segs[10]

    index = 1
    p_count = 1
    iteration = 4
    l_path = t.get_elements()
    for l in l_path:
        l_swc.append([index,p_count,l.w,l.h,l.d,1,index+1])
        index+=1
    p_count+=1

    index += 1

    t = t.parent
    while(t):
        if p_count > iteration:
            break
        l_path = t.get_elements()
        for l in l_path:
            l_swc.append([index,p_count,l.w,l.h,l.d,1,index+1])
            index+=1
        t = t.parent
        p_count+=1
        index+=1

    index+=1

    

    l_swc = np.asarray(l_swc)
    l_x = l_swc[:, 2].copy()
    l_y = l_swc[:, 3].copy()
    l_swc[:, 2] = l_y
    l_swc[:, 3] = l_x
    saveswc('test/crop1/complete_seg.swc', l_swc)

def getParent(target,index,l_swc):
    while(target.parent.size != 0):
        l_path = target.get_elements()
        for l in l_path:
            l_swc.append([index,3,l.w,l.h,l.d,1,index+1])
            index+=1
        for p in target.parent:
            getParent(p,index)

def longest_segment(topo_dists, topo_leafs,alive,leaf_nodes,topo_segs):
    # seg_swc = []
    # l_swc = []
    # f = np.argmax(topo_dists)
    # leaf = topo_leafs[f]
    # index = 0
    # while (leaf.parent):
    #     seg_swc.append([index, 3, leaf.w, leaf.h, leaf.d, 1, index + 1])
    #     l = topo_leafs[leaf.index]
    #     d = topo_dists[leaf.index]
    #     # l_swc.append([index, 3, l.w, l.h, l.d, 1, index + 1])
    #     if l != topo_leafs[f]:
    #         print('topo_leaf: ',l.w,l.h,l.d,'node: ',leaf.w,leaf.h,leaf.d,'dst: ',d)
    #     leaf = leaf.parent
    #     index += 1
    #     # print(topo_dists[])
    # seg_swc.append([index, 3, leaf.w, leaf.h, leaf.d, 1, index + 1])
    # # l_swc.append([index, 3, l.w, l.h, l.d, 1, index + 1])
    # # index = 0
    # # leaf = None
    # # for i in topo_leafs:
    # #     if i is not None and i.w == 13 and i.h==7 and i.d == 37:
    # #         leaf = i


    # # seg_swc = np.asarray(seg_swc)
    # # swc_x = seg_swc[:, 2].copy()
    # # swc_y = seg_swc[:, 3].copy()
    # # seg_swc[:, 2] = swc_y
    # # seg_swc[:, 3] = swc_x
    # # saveswc('test/crop1/longest_seg2.swc', seg_swc)

    # l_swc = np.asarray(seg_swc)
    # l_x = l_swc[:, 2].copy()
    # l_y = l_swc[:, 3].copy()
    # l_swc[:, 2] = l_y
    # l_swc[:, 3] = l_x
    # saveswc('test/crop1/l_seg.swc', l_swc)


    l_swc = []
    sort_segs = []
    for seg in topo_segs:
        sort_segs.append(seg)

    # sort by the length of the segment
    sort_segs.sort(key=lambda x:x.length, reverse=True)
    sort_segs = np.asarray(sort_segs)
    l_path = sort_segs[10].get_elements()
    print('longest segs: ',sort_segs[0].length,'size: ',l_path.size)
    index = 1
    for l in l_path:
        l_swc.append([index,3,l.w,l.h,l.d,1,index+1])
        index+=1

    l_swc = np.asarray(l_swc)
    l_x = l_swc[:, 2].copy()
    l_y = l_swc[:, 3].copy()
    l_swc[:, 2] = l_y
    l_swc[:, 3] = l_x
    saveswc('test/crop1/l_seg.swc', l_swc)