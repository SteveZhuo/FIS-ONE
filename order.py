import os.path

import numpy as np
from itertools import permutations
from sklearn.metrics import pairwise_distances, jaccard_score
from pyjarowinkler import distance

from util import *
from k_opt_tsp import tsp_2_opt

def get_floor_order_str(floors):
    id_list = list(set(range(len(floors))))
    first_id = np.min(id_list)
    id_str_list = [str(i-first_id) for i in id_list]
    return "".join(id_str_list)

def id_str_order(floors):
    id_list = list(set(range(len(floors))))
    first_id = np.min(id_list) 
    id_str_list = [str(i-first_id) for i in id_list]
    return "".join(id_str_list), "".join(id_str_list[::-1])

def floor_str_order(floors):
    floor_id = []
    for floor in floors:
        floor_id.append(floor_map[floor])

    first_id = np.min(floor_id)
    sorted_ids = [i-first_id for i in floor_id]
    sorted_ids.sort()
    floor_ids = [str(sorted_ids.index(i-first_id)) for i in floor_id]
    return "".join(floor_ids)

def hit(t_str, f_str):
    return t_str==f_str

def jaccard_distance(cluster_i, cluster_j):
    return jaccard_score(np.sum(cluster_i, axis = 0)!=0, np.sum(cluster_j, axis = 0)!=0)

def adapted_jaccard_distance(cluster_i, cluster_j):
    cluster_i_e = []
    cluster_j_e = []
    for cluster in np.sum(cluster_i, axis = 0):
        if cluster!=0:
            cluster_i_e.append(cluster)
    for cluster in np.sum(cluster_j, axis = 0):
        if cluster!=0:
            cluster_j_e.append(cluster)
    f_share = np.dot(np.sum(cluster_i, axis = 0), np.sum(cluster_j, axis = 0))
    f_diff_i = np.dot(np.sum(cluster_i, axis = 0)==0, np.sum(cluster_j, axis = 0))*np.average(cluster_i_e)
    f_diff_j = np.dot(np.sum(cluster_j, axis = 0)==0, np.sum(cluster_i, axis = 0))*np.average(cluster_j_e)
    f_diff = f_diff_i+f_diff_j
    return f_share/(f_share+f_diff)

def calculate_dist_statistics(dists):
    return {
        # "min": np.min(dists[dists != 0]),
        "min": np.min(np.min(dists),0),
        "max": np.max(dists),
        "median": np.median(dists),
        "mean": np.mean(dists)
    }

def show_cluster_pair_statistics(cluster_i, cluster_j, args):
    if args.order_id==0: # pairwise
        inter_dists = pairwise_distances(cluster_i, cluster_j)
    elif args.order_id==1: # AP, jaccard
        inter_dists = jaccard_distance(cluster_i, cluster_j)
    else: # AP, adapted jaccard
        inter_dists = adapted_jaccard_distance(cluster_i, cluster_j)
    inter_stats = calculate_dist_statistics(inter_dists)
    return inter_stats
    
def show_cluster_statistics(cluster, args):
    return show_cluster_pair_statistics(cluster, cluster, args)

def show_result(dataset, floors, floor_labels, args, building):
    # intra-cluster
    # for tmp_floor in floors:
    #     tmp_floor_dataset = [d for d, l in zip(dataset, floor_labels) if l == tmp_floor]
    #     print(show_cluster_statistics(tmp_floor_dataset, args))
        

    # inter-cluster
    selected_dist_matrix = np.zeros((len(floors), len(floors)))
    if args.order_id==0: # pairwise
        for i, floor_i in enumerate(floors):
            floor_i_dataset = [d for d, l in zip(dataset, floor_labels) if l == floor_i]
            for j in range(i+1, len(floors)):
                floor_j = floors[j]
                # print(f"Floor {floor_i} and Floor {floor_j}:")
                
                floor_j_dataset = [d for d, l in zip(dataset, floor_labels) if l == floor_j]
                inter_stats = show_cluster_pair_statistics(floor_i_dataset, floor_j_dataset, args)
                selected_dist_matrix[i,j] = inter_stats["median"]
                selected_dist_matrix[j,i] = inter_stats["median"]
                # print(inter_stats)
    else: # jaccard
        for i, floor_i in enumerate(floors):
            floor_i_dataset = [d for d, l in zip(get_ap_obs(args), floor_labels) if l == floor_i]
            for j in range(i+1, len(floors)):
                floor_j = floors[j]
                # print(f"Floor {floor_i} and Floor {floor_j}:")
                floor_j_dataset = [d for d, l in zip(get_ap_obs(args), floor_labels) if l == floor_j]
                inter_stats = show_cluster_pair_statistics(floor_i_dataset, floor_j_dataset, args)
                selected_dist_matrix[i,j] = inter_stats["median"]
                selected_dist_matrix[j,i] = inter_stats["median"]
    # print(selected_dist_matrix)
    dists = []
    perms = []
    keys = set()
    topk = args.k
    for perm in permutations(range(len(floors))):
        perm_key = "".join([str(item) for item in perm])
        reverse_key = "".join([str(item) for item in perm[::-1]])
        if perm_key in keys or reverse_key in keys:
            continue
        sum_dist = 0
        for (i,j) in zip(perm[:-1], perm[1:]):
            sum_dist += selected_dist_matrix[i,j]
        dists.append(sum_dist)
        perms.append(perm)
        keys.add(perm_key)

    o1, o2 = id_str_order(floors)
    if args.order_id==0: # pairwise, want minimum
        sort_indicies = np.argsort(dists)
        for i in range(len(sort_indicies)):
            floor_order = [floors[k] for k in perms[sort_indicies[i]]]
            order_dist = dists[sort_indicies[i]]
            d1 = distance.get_jaro_distance(o1, floor_str_order(floor_order), winkler=True, scaling=0.1)
            d2 = distance.get_jaro_distance(o2, floor_str_order(floor_order), winkler=True, scaling=0.1)
            # print(f"perm detected: {floor_order}, with min_dist: {order_dist}")
            if i<topk:
                print(f"perm detected: {floor_order}, with edit distance from true order: {max(d1, d2)}")
            if hit(o1, floor_str_order(floor_order)) or hit(o2, floor_str_order(floor_order)):
                print("correct sequence detected at {}".format(i+1))
                break
    else: # jaccard, want maximum
        sort_indicies = np.flipud(np.argsort(dists))
        for i in range(len(sort_indicies)):
            floor_order = [floors[k] for k in perms[sort_indicies[i]]]
            order_dist = dists[sort_indicies[i]]
            d1 = distance.get_jaro_distance(o1, floor_str_order(floor_order), winkler=True, scaling=0.1)
            d2 = distance.get_jaro_distance(o2, floor_str_order(floor_order), winkler=True, scaling=0.1)
            if i<topk:
                print(f"perm detected: {floor_order}, with edit distance from true order: {max(d1, d2)}")
            if hit(o1, floor_str_order(floor_order)) or hit(o2, floor_str_order(floor_order)):
                print("correct sequence detected at {}".format(i+1))
                break

def show_result_2_opt(dataset, floors, floor_labels, args, building):
    selected_dist_matrix = np.zeros((len(floors), len(floors)))
    if args.order_id == 0:  # pairwise
        for i, floor_i in enumerate(floors):
            floor_i_dataset = [d for d, l in zip(dataset, floor_labels) if l == floor_i]
            for j in range(i + 1, len(floors)):
                floor_j = floors[j]
                # print(f"Floor {floor_i} and Floor {floor_j}:")

                floor_j_dataset = [d for d, l in zip(dataset, floor_labels) if l == floor_j]
                inter_stats = show_cluster_pair_statistics(floor_i_dataset, floor_j_dataset, args)
                selected_dist_matrix[i, j] = inter_stats["median"]
                selected_dist_matrix[j, i] = inter_stats["median"]
                # print(inter_stats)
    else:  # jaccard
        for i, floor_i in enumerate(floors):
            floor_i_dataset = [d for d, l in zip(get_ap_obs(args), floor_labels) if l == floor_i]
            for j in range(i + 1, len(floors)):
                floor_j = floors[j]
                floor_j_dataset = [d for d, l in zip(get_ap_obs(args), floor_labels) if l == floor_j]
                inter_stats = show_cluster_pair_statistics(floor_i_dataset, floor_j_dataset, args)
                selected_dist_matrix[i, j] = 1 / (inter_stats["median"])
                selected_dist_matrix[j, i] = 1 / (inter_stats["median"])
    order_2_opt = tsp_2_opt(selected_dist_matrix, list(range(len(floors))))
    floor_pred = "".join([str(item) for item in order_2_opt])
    o1  = get_floor_order_str(floors)
    dist = distance.get_jaro_distance(o1, floor_pred, winkler=True, scaling=0.1)
    print (f"edit distance: {dist}")

    return order_2_opt