import os
import os.path as osp
import pickle5 as pickle

import random
import numpy as np
from collections import Counter

# floor scaling as we have multiple floor labelling systems
floor_map = {"B3": -3, "B2": -2, "B1": -1, "F1": 0, "F2": 1, "F3": 2, "F4": 3, "F5": 4, "F6": 5, "F7": 6, "F8": 7, "F9": 8, "F10": 9, "F11": 10, "F12": 11,
             "3B": -3, "2B": -2, "1B": -1, "1F": 0, "2F": 1, "3F": 2, "4F": 3, "5F": 4, "6F": 5, "7F": 6, "8F": 7, "9F": 8, "10F": 9, "11F": 10, "12F": 11,
             "LG3": -3, "LG2": -2, "LG1": -1, "L1": 0, "L2": 1, "L3": 2, "L4": 3, "L5": 4, "L6": 5, "L7": 6, "L8": 7, "L9": 8, "L10": 9, "L11": 10, "L12": 11,
             "B": -2, "G": -1, "LM": 1, "P1": -3, "P2": -4, "BM": -2, "MF": 13, "LG": -1,
             "BF": -1, "GF":-1, 'GMF':-3, '1MF':-2, 'MTR':-1, 'B2F':-3, 'B1F':-2, '2':2}

# get dictionary of floors of each obs
def get_floors(args):
    output_path = osp.abspath(osp.join(os.getcwd(), args.output_path, args.site_id))
    floor_path = osp.abspath(osp.join(output_path, args.floor_path))
    with open(floor_path, 'r') as f_in:
        lines = f_in.read()
        floor_dict = eval(lines)
    return floor_dict

# get observation of dim of obs, whose floor's occurrence is larger than threhold
def get_embs_with_thresh(args):
    output_path = osp.abspath(osp.join(os.getcwd(), args.output_path, args.site_id))
    emb_path = osp.abspath(osp.join(output_path, args.emb_path.format(args.dim)))
    # emb_path = osp.abspath(osp.join(os.getcwd(), args.compare_path, args.site_id, args.emb_path.format(args.dim)))
    vectors = []
    rows = []
    u_count = 0

    floor_data = get_floors(args)
    floor_counter = Counter(floor_data.values())
    new_floor_data = {}
    
    with open(emb_path, 'r') as f_in:
        lines = f_in.readlines()
        for line in lines[1:]:
            element_list = line.split()
            node_type = element_list[0][0]
            coordinate_list = [float(item) for item in element_list[1:]]


            if node_type == "u":
                if floor_counter[floor_data[element_list[0]]]>args.threshold: # to ignore threhold (i.e. use data even when its insignificant, delete this line)
                    vectors.insert(0, coordinate_list)
                    rows.insert(0, element_list[0])
                    u_count += 1
                    new_floor_data[element_list[0]] = floor_data[element_list[0]]
            else:
                vectors.append(coordinate_list)
                rows.append(element_list[0])

        dataset = np.array(vectors)
    
    return dataset, rows, u_count, new_floor_data

# get the set of all aps
def get_aps(args):
    output_path = osp.abspath(osp.join(os.getcwd(), args.output_path, args.site_id))
    ap_path = osp.abspath(osp.join(output_path, args.ap_path.format(args.site_id)))

    ap_set = []
    with open(ap_path, 'rb') as f_in:
        series_ap = pickle.load(f_in)
        for ap in series_ap:
            ap_set.append(ap[1])
    
    return ap_set

# get ap occurrence for each obs
def get_ap_obs(args):
    aps = get_aps(args)
    output_path = osp.abspath(osp.join(os.getcwd(), args.output_path, args.site_id))
    obs_path = osp.abspath(osp.join(output_path, args.obs_path.format(args.site_id)))
    with open(obs_path, 'rb') as f_in:
        series_obs = pickle.load(f_in)
        ap_obs = np.zeros((len(series_obs), len(aps)))
        i = 0
        for obs in series_obs:
            for ap in list(obs[3].keys()):
                if(obs[3][ap]>args.thresh):
                    ap_obs[i][aps.index(ap)] += 1
            i += 1
    return ap_obs

# get ap occurrence for each floor
def print_ap_floors(args):
    floor_dict = get_floors(args)
    floor_list = list(set(floor_dict.values()))
    ap_occurrence_dict = {}
    aps = get_aps(args)
    output_path = osp.abspath(osp.join(os.getcwd(), args.output_path, args.site_id))
    obs_path = osp.abspath(osp.join(output_path, args.obs_path.format(args.site_id)))
    with open(obs_path, 'rb') as f_in:
        series_obs = pickle.load(f_in)
        ap_floors = np.zeros((len(floor_list), len(aps)))
        for obs in series_obs:
            for ap in list(obs[3].keys()):
                ap_floors[floor_list.index(floor_dict[obs[0]])][aps.index(ap)] += 1

    ap_occurrence_list = np.sum(ap_floors>0, axis=0)
    for ap_occurrence in ap_occurrence_list:
        if ap_occurrence not in ap_occurrence_dict.keys():
            ap_occurrence_dict[ap_occurrence] = 1
        else:
            ap_occurrence_dict[ap_occurrence] += 1
    # print("Floors: ", floor_list)
    # print("#APs for each floor: ", np.sum(ap_floors>0, axis=1))
    # print("Count of #Floors of AP occurrence: ", ap_occurrence_dict)
    floor_dict = {}
    for i in range(len(floor_list)):
        floor_dict[floor_list[i]] = int(np.sum(ap_floors>0, axis=1)[i])
    # print("#APs for each floor: ", floor_dict)
    # print("Count of #Floors of AP occurrence: ", ap_occurrence_dict)
    return floor_dict, ap_occurrence_dict

def load_file_comparison(filename, observation_ids, observation_sign, mac_all, file_type, prune):
    max_id = 0
    if observation_ids:
        max_id = max(max_id, int(observation_ids[-1][0][1:]))

    with open(filename, "r") as f_in:
        while True:
            line = f_in.readline().rstrip(" \n")
            if not line:
                break
            if file_type == "path":
                if line.startswith("Start:"):
                    device_id = os.path.basename(filename)[:2]
                    breakpoints = [[float(coor) for coor in item.split(
                        ",")] for item in f_in.readline().rstrip(" \n").split(" ")]
                    timestamps = [
                        int(ts) for ts in f_in.readline().rstrip(" \n").split(" ")]
                    continue
                timestamp, rssi_pairs = line.split(" ", 1)
                ground_truth = interpolate_point(
                    int(timestamp), timestamps, breakpoints)
            elif file_type == "db":
                device_id = os.path.basename(filename)
                coor, rssi_pairs = line.split(" ", 1)
                ground_truth = [float(item) for item in coor.split(",")]
            elif file_type == "new":
                wifi_json = json.loads(line)
                timestamp = wifi_json['sysTimeMs']
                rssi_pairs = ''
                for item in wifi_json['data']:
                    rssi_pairs += str(item['bssid'].replace(':','')) + ',' + str(item['rssi']) + ' '
                rssi_pairs = rssi_pairs.strip(' ')
                ground_truth = [None,None]
                device_id = None
            rssi_dict = {}
            for rssi_pair in rssi_pairs.split(" "):
                mac = rssi_pair.split(",")[0]
                rssi = rssi_pair.split(",")[1]
                # remove virtual mac
                if prune and is_virtual_mac(mac):
                    continue
                rssi_dict[mac] = float(rssi)
                if mac not in mac_all:
                    mac_all.append(mac)
            if rssi_dict:
                observation_ids.append(
                    ["{}{}".format(observation_sign, max_id+1), device_id, ground_truth, rssi_dict])
                max_id += 1

    print("{} loaded".format(filename))
    return observation_ids, mac_all

def load_file_comparison(filename, observation_ids, observation_sign, mac_all, file_type, prune):
    max_id = 0
    if observation_ids:
        max_id = max(max_id, int(observation_ids[-1][0][1:]))

    with open(filename, "r") as f_in:
        while True:
            line = f_in.readline().rstrip(" \n")
            if not line:
                break
            if file_type == "path":
                if line.startswith("Start:"):
                    device_id = os.path.basename(filename)[:2]
                    breakpoints = [[float(coor) for coor in item.split(
                        ",")] for item in f_in.readline().rstrip(" \n").split(" ")]
                    timestamps = [
                        int(ts) for ts in f_in.readline().rstrip(" \n").split(" ")]
                    continue
                timestamp, rssi_pairs = line.split(" ", 1)
                ground_truth = interpolate_point(
                    int(timestamp), timestamps, breakpoints)
            elif file_type == "db":
                device_id = os.path.basename(filename)
                coor, rssi_pairs = line.split(" ", 1)
                ground_truth = [float(item) for item in coor.split(",")]
            elif file_type == "new":
                wifi_json = json.loads(line)
                timestamp = wifi_json['sysTimeMs']
                rssi_pairs = ''
                for item in wifi_json['data']:
                    rssi_pairs += str(item['bssid'].replace(':','')) + ',' + str(item['rssi']) + ' '
                rssi_pairs = rssi_pairs.strip(' ')
                ground_truth = [None,None]
                device_id = None
            rssi_dict = {}
            for rssi_pair in rssi_pairs.split(" "):
                mac = rssi_pair.split(",")[0]
                rssi = rssi_pair.split(",")[1]
                # remove virtual mac
                if prune and is_virtual_mac(mac):
                    continue
                rssi_dict[mac] = float(rssi)
                if mac not in mac_all:
                    mac_all.append(mac)
            if rssi_dict:
                observation_ids.append(
                    ["{}{}".format(observation_sign, max_id+1), device_id, ground_truth, rssi_dict])
                max_id += 1

    print("{} loaded".format(filename))
    return observation_ids, mac_all


def series_to_file(obj, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out, -1)
        print("Data written into {}".format(filename))


def file_to_series(filename):
    with open(filename, 'rb') as f_in:
        series = pickle.load(f_in)
        print("File {} loaded.".format(filename))
        return series


def rssi2weight(offset, rssi):
    return offset + rssi


def is_virtual_mac(mac_addr):
    mac_addr = mac_addr.replace(":", "").upper()
    first_hex = int(mac_addr[0:2], 16)
    return first_hex & 0x02 != 0


def interpolate_point(timestamp, timestamps, breakpoints):
    if timestamp <= timestamps[0]:
        print("timestamp too small: {} <= {}".format(timestamp, timestamps[0]))
        return breakpoints[0]
    if timestamp >= timestamps[-1]:
        print("timestamp too large: {} >= {}".format(
            timestamp, timestamps[-1]))
        return breakpoints[-1]

    for idx in range(len(timestamps)-1):
        if timestamps[idx] <= timestamp <= timestamps[idx+1]:
            return [breakpoints[idx][coor_id] + (timestamp - timestamps[idx]) /
                    (timestamps[idx+1] - timestamps[idx]) *
                    (breakpoints[idx+1][coor_id] - breakpoints[idx][coor_id])
                    for coor_id in [0, 1]]    

def get_index(lst, item):
    return [i for i in range(len(lst)) if lst[i] == item]