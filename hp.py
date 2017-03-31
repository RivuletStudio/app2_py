import numpy as np
import math
from node import *
from utils.io import *
from random import randint

"""
hierarchical pruning based on the initial tree reconstrucionn by fast marching

"""
def hp(img,bimg,size,alive,out,threshold):

    filter_segs = swc2topo_segs(img,size,alive,out,threshold)
    
    # calculate radius for every node
    print('--calculating radius for every node')
    index = 0
    for seg in filter_segs:
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
    seg_swc = np.asarray(seg_swc)
    swc_x = seg_swc[:, 2].copy()
    swc_y = seg_swc[:, 3].copy()
    seg_swc[:, 2] = swc_y
    seg_swc[:, 3] = swc_x
    saveswc(out+'length_threshold5_test.swc', seg_swc)


    print('--Hierarchical Pruning')
    result_segs = hierchical_coverage_prune(filter_segs,img,out)
    seg_swc = []
    temp_swc = []

    count = 0
    index = 0

    for seg in result_segs:
        seg_tree = seg.get_elements()
        # longest_segment(seg_tree,index,out)
        color = randint(1,200)
        temp_swc = []
        for i in seg_tree:
            if i.parent is None:
                seg_swc.append([i.index, color, i.w, i.h, i.d, i.radius, -1])
                count += 1
            else:
                seg_swc.append([i.index, color, i.w, i.h, i.d, i.radius, i.parent.index])

        # print(index)
        index+=1

    seg_swc = np.asarray(seg_swc)
    swc_x = seg_swc[:, 2].copy()
    swc_y = seg_swc[:, 3].copy()
    seg_swc[:, 2] = swc_y
    seg_swc[:, 3] = swc_x
    saveswc(out+'new1.swc', seg_swc)

    return

"""
build segments based on the swc from the initial reconstruction

"""
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

    # print('leaf_size: ', leaf_nodes.size)

    # print('child no of index 31: ', child_no[0])

    count = 0
    for i in alive:
        if i.parent is None:
            continue
        elif i.parent.index == 0:
            # print(i.w,i.h,i.d)
            count += 1
    # print('child no of index 31: ', count)

    leaf_num = leaf_nodes.size




    # furthest leaf distance for each tree node
    topo_dists = np.zeros(tol_num)
    topo_leafs = np.empty(tol_num, dtype=spatial)

        
    for leaf in leaf_nodes:
        child_node = leaf
        parent_node = child_node.parent
        cid = child_node.index
        topo_leafs[cid] = leaf
        topo_dists[cid] = img[leaf.w][leaf.h][leaf.d] / 255.0
        while (parent_node):

            pid = parent_node.index
            tmp_dst = img[parent_node.w][parent_node.h][
                parent_node.d] / 255.0 + topo_dists[cid]

            if (tmp_dst >= topo_dists[pid]):
                topo_dists[pid] = tmp_dst
                topo_leafs[pid] = topo_leafs[cid]
            else:
                break

            child_node = parent_node
            cid = pid
            parent_node = parent_node.parent

    fp = np.argmax(topo_dists)
    fn = topo_leafs[fp]
    # print('furthest point location: ', fn.w, fn.h, fn.d, 'index: ',fp,'length: ',
          # topo_dists[fp])
    # print('seed topo to this point: ', topo_leafs[0].w, topo_leafs[0].h,
          # topo_leafs[0].d, topo_dists[0])

    topo_segs = np.empty(leaf_num, dtype=segment)

    index = 0
    for leaf in leaf_nodes:     
        root_marker = leaf
        root_parent = root_marker.parent
        level = 1

        while (root_parent and topo_leafs[root_parent.index] == leaf):
            if child_no[root_marker.index] >= 2:
                level += 1
            root_marker = root_parent
            root_parent = root_marker.parent

        dst = topo_dists[root_marker.index]
        topo_seg = segment(leaf, root_marker, dst, level)
        topo_segs[index] = topo_seg

        if (root_parent is None):
            topo_seg.parent = None
        else:
            leaf_marker2 = topo_leafs[root_parent.index]
            loc = np.argwhere(leaf_nodes == leaf_marker2)
            topo_seg.parent = topo_segs[loc[0][0]]
        index += 1

    # complete_segment(topo_dists, topo_leafs,alive,leaf_nodes,topo_segs,out)

    filter_segs = np.array([])
    # print('Current Segments size:  ',topo_segs.size)
    # print('--Prune by length threhold')

    for seg in topo_segs:
        # seg_length = np.append(seg_length,seg.length)
        if seg.length > 4:
            filter_segs = np.append(filter_segs, seg)

    # print('Current Segments size:  ',filter_segs.size)
    # for i in topo_segs:
    #     if(i.parent is None):
            # print('lolxx')

    # for i in filter_segs:
    #     print(type(i.parent))
    return filter_segs

"""
hierchical coverage pruning based on the segment reconstruction.
segments with coverage ratio less than threshold will be pruned

"""
def hierchical_coverage_prune(filter_segs,img,out):
    sort_segs = []
    for seg in filter_segs:
        sort_segs.append(seg)

    tmpimg = img.copy()
    bb = np.zeros(img.shape)
    sort_segs.sort(key=lambda x:x.length, reverse=True)
    sort_segs = np.asarray(sort_segs)
    result_segs = []
    delete_segs = []

    sort_index = 0
    seg_index = 0
    for seg in sort_segs:
        current = seg.leaf
        root = seg.root
        overlap = 0
        non_overlap = 0
        tol_num = 0 # Total number of the covered area 

        while (current != root):
            r = math.ceil(current.radius * 1.5)
            x, y, z = np.meshgrid(
                    constrain_range(current.w - r, current.w + r + 1, 0, img.shape[0]),
                    constrain_range(current.h - r, current.h + r + 1, 0, img.shape[1]),
                    constrain_range(current.d - r, current.d + r + 1, 0, img.shape[2]))
            overlap += bb[x, y, z].sum()
            tol_num += x.shape[0] * x.shape[1] * x.shape[2] 
            current = current.parent
        
        coverage = overlap / tol_num
        
        # if sort_index == 79 or sort_index == 141 or sort_index == 154 or sort_index == 229:
            # print("== Seg coverage:", overlap, tol_num,coverage)
        if (coverage < 0.5):
            result_segs.append(seg)
            seg_index+=1
            store_segment(seg,seg_index,sort_index,out)
        else:
            delete_segs.append(seg)

        current = seg.leaf
        root = seg.root
        overlap = 0
        non_overlap = 0

        while (current != root):
            x,y,z = np.meshgrid(
                constrain_range(current.w - r, current.w + r + 1, 0, img.shape[0]),
                constrain_range(current.h - r, current.h + r + 1, 0, img.shape[1]),
                constrain_range(current.d - r, current.d + r + 1, 0, img.shape[2]))            
            bb[x, y, z] = 1
            current = current.parent
        sort_index+=1
    filtered_result_segs = result_segs  
    # print(len(filtered_result_segs))
    # filtered_result_segs = []

    # TODO: Not sure if it works!
    # # Delete the added segments whose parents were deleted
    # for seg in result_segs:
    #     if seg.parent not in delete_segs:
    #         filtered_result_segs.append(seg)       

    return filtered_result_segs

"""
make sure the mask area is in the bound of the image

"""
def constrain_range(min, max, minlimit, maxlimit):
    return list(
        range(min if min > minlimit else minlimit, max
              if max < maxlimit else maxlimit))


"""
estimate the radius for each node (PHC)

"""
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

"""
estimate the radius for each node (Siqi)

"""
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

"""
test method for store a complete segment

"""
def complete_segment(topo_dists, topo_leafs,alive,leaf_nodes,topo_segs,out):
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
    saveswc(out+'complete_seg.swc', l_swc)

"""
test method for store the longest segment

"""
def longest_segment(l_path,index,out):
    l_swc = []
    # sort_segs = []
    # for seg in topo_segs:
    #     sort_segs.append(seg)

    # sort by the length of the segment
    # sort_segs.sort(key=lambda x:x.length, reverse=True)
    # sort_segs = np.asarray(sort_segs)
    # l_path = sort_segs[1].get_elements()
    # print('longest segs: ',sort_segs[1].length,'size: ',l_path.size)
    index = 1
    for l in l_path:
        l_swc.append([index,3,l.w,l.h,l.d,1,index+1])
        index+=1

    l_swc = np.asarray(l_swc)
    l_x = l_swc[:, 2].copy()
    l_y = l_swc[:, 3].copy()
    l_swc[:, 2] = l_y
    l_swc[:, 3] = l_x
    saveswc(out+'l_seg'+str(index)+'.swc', l_swc)


def store_segment(seg,seg_index,sort_index,out):
    path = seg.get_elements()
    color = randint(1,200)
    temp_swc = []
    for l in path:
        if l.parent is None:
            temp_swc.append([l.index,color,l.w,l.h,l.d,1,-1])
        else:
            temp_swc.append([l.index,color,l.w,l.h,l.d,1,l.parent.index])
    temp_swc = np.asarray(temp_swc)
    temp_x = temp_swc[:, 2].copy()
    temp_y = temp_swc[:, 3].copy()
    temp_swc[:, 2] = temp_y
    temp_swc[:, 3] = temp_x
    saveswc(out+'seg_'+str(seg_index)+'_sorted'+str(sort_index)+'.swc',temp_swc)

