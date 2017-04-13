import numpy as np
import math
from node import *
from utils.io import *
from random import randint

"""
hierarchical pruning based on the initial tree reconstrucionn by fast marching

"""
def hp(img,bimg,size,alive,out,threshold):

    # alive = loadswc('test/2000-1/new_fm_ini.swc')
    print(alive[27])
    print(type(alive))
    swc_x = alive[:, 2].copy()
    swc_y = alive[:, 3].copy()
    alive[:, 2] = swc_y
    alive[:, 3] = swc_x
    # saveswc(out + 'load_fm_ini.swc',alive)
    # alive[:,2] = alive[:,2] + 1
    # alive[:,3] = alive[:,3] + 1
    # alive[:,4] = alive[:,4] + 1
    saveswc(out + 'load_fm_ini.swc',alive)
    filter_segs = swc2topo_segs(img,size,alive,out,threshold)
    
    # calculate radius for every node
    # store filter_segs [leaf_index,root_index,parent_index,length,level]
    print('--calculating radius for every node')

    for seg in filter_segs:
        leaf_marker = seg[0]
        root_marker = seg[1]
        p = int(leaf_marker)
        while(1):
            real_threshold = 40
            if (real_threshold < threshold):
                real_threshold = threshold
            alive[p][5] = getradius(bimg,alive[p][2],alive[p][3],alive[p][4])
            if (p == root_marker):
                break
            p = int(alive[p][6])


    print('--Hierarchical Pruning')
    result_segs = hierchical_coverage_prune(alive,filter_segs,img,out)
    return

"""
build segments based on the swc from the initial reconstruction

"""
def swc2topo_segs(img,size,alive,out,threshold):
    tol_num = alive.shape[0]
    print('27',alive[27])

    leaf_nodes = np.array([])
    child_no = np.zeros(tol_num, dtype=int)

    for i in alive[1:tol_num-1:,0]:
        child_no[int(alive[int(i)][6])] += 1
        # child_no[i] = 
        

    # calculate distance for every tree nodes
    child_index = np.where(child_no == 0)
    leaf_nodes = alive[child_index]
    for leaf in leaf_nodes:
        if leaf[2] == 71 and leaf[3] == 298 and leaf[4] == 77:
            print('found!')

    print('leaf size',leaf_nodes.shape)


    topo_dists = np.zeros(tol_num)
    topo_leafs = np.empty(tol_num)

    child_node = leaf_nodes[11]

    parent_node = alive[int(child_node[6])]
    # print(parent_node)
    # print(int(child_node[0]))
    # # print(img[child_node[2]][child_node[3]][child_node[4]])
    # print(img[116][210][7])

    for leaf in leaf_nodes:
        child_node = leaf
        parent_node = alive[int(child_node[6])]
        cid = int(child_node[0])
        topo_leafs[cid] = leaf[0]
        # print(img[2][3][4]/255.0)
        topo_dists[cid] = img[leaf[2]][leaf[3]][leaf[4]]/255.0
        while parent_node[0] != 0:
            pid = int(parent_node[0])
            tmp_dst = img[int(parent_node[2])][int(parent_node[3])][int(parent_node[4])]/255.0 + topo_dists[cid]
            # print(tmp_dst)

            if (tmp_dst >= topo_dists[pid]):
                topo_dists[pid] = tmp_dst
                topo_leafs[pid] = topo_leafs[cid]
            else:
                break
            child_node = parent_node
            cid = pid
            parent_node = alive[int(parent_node[6])]


    # print(topo_dists[11])
    fp = np.argmax(topo_dists)
    print(fp)
    fn = topo_leafs[fp]
    print(fn)
    print(topo_dists[fp])
    print('furthest point location: ', alive[int(fn)][2], alive[int(fn)][3], alive[int(fn)][4], 'index: ',fp,'length: ',
          topo_dists[fp])
    print('seed topo to this point: ', alive[int(topo_leafs[0])][2], alive[int(topo_leafs[0])][3],
          alive[int(topo_leafs[0])][4], topo_dists[0])
    # # topo_segs (leaf,root,parent,dst,level)
    # print(np.argwhere(topo_dists==0).shape)
    topo_segs = np.array([[]])

    
    # store filter_segs [leaf_index,root_index,parent_index,length,level]
    for leaf in leaf_nodes:
        root_marker = leaf
        root_parent = alive[int(root_marker[6])]
        level = 1

        # print('root id',root_parent[0])
        # print('topo tp',topo_leafs[root_parent[0]])
        # print('leaf id',leaf[0])

        while (root_parent[0] != 0 and topo_leafs[int(root_parent[0])] == leaf[0]):
            if child_no[int(root_marker[0])] >= 2:
                level += 1
            root_marker = root_parent
            root_parent = alive[int(root_marker[6])]
            # print('bingo')
        dst = topo_dists[int(root_marker[0])]

        parent_index = topo_leafs[int(root_parent[0])]

        if topo_segs.size == 0:
            topo_segs = np.asarray([[leaf[0],root_marker[0],parent_index,dst,level]])
        else:
            topo_segs = np.vstack((topo_segs,[leaf[0],root_marker[0],parent_index,dst,level]))
        # print(topo_segs.shape)

    # print(topo_segs)
    # print(topo_segs[:,3])
    filter_segs = topo_segs[np.argwhere(topo_segs[:,3] > 5)]
    filter_segs = np.squeeze(filter_segs, axis=(1,))
    # print(filter_segs)
    print('filter_segs',filter_segs.shape)

    # store filter segments
    out_swc = np.array([[]])

    for i in filter_segs:
        current = int(i[0])
        while(current != i[1]):
            if out_swc.size == 0:
                out_swc = np.array(alive[current])
            else:
                out_swc = np.vstack((out_swc,alive[current]))       
            current = alive[current][6]

    swc_x = out_swc[:, 2].copy()
    swc_y = out_swc[:, 3].copy()
    out_swc[:, 2] = swc_y
    out_swc[:, 3] = swc_x
    saveswc(out + 'new_filter_segments.swc',out_swc)




    return filter_segs

"""
hierchical coverage pruning based on the segment reconstruction.
segments with coverage ratio less than threshold will be pruned

"""
def hierchical_coverage_prune(alive,filter_segs,img,out):
    sort_segs = filter_segs[np.argsort(filter_segs[:,3])]
    sort_segs = sort_segs[::-1]
    result_segs = np.array([[]])
    bb = np.zeros(img.shape)
    print(int(sort_segs[0][3]))
    print(sort_segs[1][3])
    print(sort_segs[2][3])

    index = 0
    # store filter_segs [leaf_index,root_index,parent_index,length,level]
    for seg in sort_segs:
        # seg = sort_segs[index]
        current = int(seg[0])
        root = int(seg[1])
        overlap = 0
        non_overlap = 0
        tol_num = 0 # Total number of the covered area 

        while (current != root):
            r = math.ceil(alive[int(current)][5] * 1.5)
            w = alive[current][2]
            h = alive[current][3]
            d = alive[current][4]
            x, y, z = np.meshgrid(
                    constrain_range(w - r, w + r + 1, 0, img.shape[0]),
                    constrain_range(h - r, h + r + 1, 0, img.shape[1]),
                    constrain_range(d - r, d + r + 1, 0, img.shape[2]))
            overlap += bb[x, y, z].sum()
            tol_num += x.shape[0] * x.shape[1] * x.shape[2] 
            current = int(alive[current][6])
        
        coverage = overlap / tol_num
        

        # if coverage smaller than threshold
        if (coverage < 3.0/4.0):
            current = int(seg[0])
            while (1):
                if result_segs.size == 0:
                    result_segs = np.array(alive[current])
                else:
                    result_segs = np.vstack((result_segs,alive[current]))
                if current == seg[1]:
                    break
                else:
                    current = alive[current][6]

        # delete this segment and all children
        # else:
            # if children_index.size != 0:
            #     for child in children_index:
            #         next_index = np.argwhere(sort_segs)
                # if exists a child, delete child and find next child





        current = int(seg[0])
        root = int(seg[1])
        overlap = 0
        non_overlap = 0

        while (current != root):
            x,y,z = np.meshgrid(
                constrain_range(w - r, w + r + 1, 0, img.shape[0]),
                constrain_range(h - r, h + r + 1, 0, img.shape[1]),
                constrain_range(d - r, d + r + 1, 0, img.shape[2]))            
            bb[x, y, z] = 1
            current = int(alive[current][6])

        # index+=1

    # delete leaf segments which the parent have already been deleted
    parent_index = result_segs[:, 6]
    delete_index = []
    index = 0
    for p in parent_index:
        if p not in result_segs[:, 0]:
            delete_index.append(index)
        index+=1
    print(result_segs.shape)
    result_segs = np.delete(result_segs,delete_index,axis=0)
    print(result_segs.shape)


    
    swc_x = result_segs[:, 2].copy()
    swc_y = result_segs[:, 3].copy()
    result_segs[:, 2] = swc_y
    result_segs[:, 3] = swc_x
    saveswc(out + 'new_result_3_4.swc',result_segs)
    return result_segs


    # print(len(filtered_result_segs))
    # filtered_result_segs = []

    # TODO: Not sure if it works!
    # # Delete the added segments whose parents were deleted
    # for seg in result_segs:
    #     if seg.parent not in delete_segs:
    #         filtered_result_segs.append(seg)       

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
    saveswc(out+'new_longest'+'.swc', l_swc)


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

