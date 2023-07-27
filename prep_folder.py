import os
import os.path as osp
import pickle
import json
import shutil

from util import *

# First step: load file and get observation points
def load_file(filename, observation_ids, observation_sign, file_type, prune, th):
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
                    device_id = osp.basename(filename)[:2]
                    breakpoints = [[float(coor) for coor in item.split(
                        ",")] for item in f_in.readline().rstrip(" \n").split(" ")]
                    timestamps = [
                        int(ts) for ts in f_in.readline().rstrip(" \n").split(" ")]
                    continue
                timestamp, rssi_pairs = line.split(" ", 1)
                ground_truth = interpolate_point(
                    int(timestamp), timestamps, breakpoints)
            elif file_type == "db":
                device_id = osp.basename(filename)
                if len(line.split(" ", 1))>1:
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
                mac = rssi_pair.split(",")[0][3:]
                rssi = rssi_pair.split(",")[1]
                # remove virtual mac
                #if prune and is_virtual_mac(mac): 
                #    continue
                if float(rssi) >= th:
                    rssi_dict[mac] = float(rssi)
            if rssi_dict:
                observation_ids.append(
                    ["{}{}".format(observation_sign, max_id+1), device_id, ground_truth, rssi_dict])
                max_id += 1

    print("{} loaded".format(filename))
    return observation_ids

# Second step: generate ap nodes from given observation points
def generate_ap_ids(ap_file, ap_sign, observation_ids):
    max_id = 0
    if osp.isfile(ap_file):
        ap_ids = file_to_series(ap_file)
        max_id = max(max_id, int(ap_ids[-1][0][1:]))
    else:
        ap_ids = []

    for observation in observation_ids:
        for mac in observation[3].keys():
            ap_flag = False
            for ap in ap_ids:
                if ap[1] == mac:
                    ap_flag = True
                    ap[3].append([observation[3][mac]]+observation[2])
                    break
            if not ap_flag:
                ap_ids.append(["{}{}".format(ap_sign, max_id+1), mac,
                               None, [[observation[3][mac]]+observation[2]]])
                max_id += 1
    if ap_ids:
        series_to_file(ap_ids, ap_file)
    else:
        print("No AP ids!")
    return ap_ids

# Third step: use observation nodes and ap nodes to generate edgelist for further use
def generate_graph_file(observation_file, ap_file, offset, output_file):    
    observation_ids = file_to_series(observation_file)
    ap_ids = file_to_series(ap_file)
    ap_dict = {}
    for ap_item in ap_ids:
        ap_dict[ap_item[1]] = ap_item[0]

    with open(output_file, "w") as f_out:
        for observation in observation_ids:
            ob_id = observation[0]
            for mac in observation[3].keys():
                #f_out.write("{} {} {}\n".format(ob_id, ap_dict[mac], rssi2weightexp(
                #    observation[3][mac])))
                f_out.write("{} {} {}\n".format(ob_id, ap_dict[mac], rssi2weight(
                    offset, observation[3][mac])))
    print("edgelist generated.")

if __name__ == "__main__":
    # load settings
    offset, ap_sign, observation_sign, th = 0, "", "", 0
    offset = 120
    ap_sign = 'i'
    observation_sign = 'u'
    th = -90
    input_dir_name = "data/input"
    output_dir_name = "data/output"
    target_dir_name = "raw_data"
    building_ids = os.listdir(input_dir_name)
    print("processing...")
    info_dict = {}
    root_dir = os.getcwd()
    input_dir = osp.join(root_dir, input_dir_name)
    output_dir = osp.join(root_dir, output_dir_name)
    # reset the output directory
    if not osp.isdir(output_dir):
        os.mkdir(output_dir)
    # else:
    #     shutil.rmtree(output_dir)
    #     os.mkdir(output_dir)

    for raw_data_dir in os.listdir(input_dir):
        if raw_data_dir not in building_ids:
            continue
        item = osp.join(input_dir, raw_data_dir)
        if osp.isdir(item):
            output_dir_dataset = osp.join(output_dir, raw_data_dir)
            target_dir = osp.join(output_dir_dataset, target_dir_name)
            if not osp.isdir(output_dir_dataset):
                os.mkdir(output_dir_dataset)
                os.mkdir(target_dir)
                os.mkdir(osp.join(output_dir_dataset, "embedding"))
                os.mkdir(osp.join(output_dir_dataset, "anchor list"))
                os.mkdir(osp.join(output_dir_dataset, "spring"))
                os.mkdir(osp.join(output_dir_dataset, "anchor map"))
                os.mkdir(osp.join(output_dir_dataset, "incremental data"))
                print (f'Output dir made {output_dir_dataset}')

            observation_file = osp.join(target_dir, "_observations_{}.pkl".format(raw_data_dir))
            ap_file = osp.join(target_dir, "_APs_{}.pkl".format(raw_data_dir))

            pruned_observation_file = osp.join(target_dir, "_observations_prune_{}.pkl".format(raw_data_dir))
            pruned_ap_file = osp.join(target_dir, "_APs_prune_{}.pkl".format(raw_data_dir))

            graph_file = osp.join(target_dir, "{}.edgelist".format(raw_data_dir))
            pruned_graph_file = osp.join(target_dir, "prune_{}.edgelist".format(raw_data_dir))

            observation_ids = []
            pruned_observation_ids = []
            for filename in os.listdir(item):
                if filename.endswith("WiFi.txt"):
                    file = osp.join(item, filename)
                    observation_ids = load_file(file, observation_ids, observation_sign, file_type="path", prune=False, th=th)
                    pruned_observation_ids = load_file(file, pruned_observation_ids, observation_sign, file_type="path", prune=True, th=th)
                elif filename.startswith("fingerprint"):
                    file = osp.join(item, filename)
                    observation_ids = load_file(file, observation_ids, observation_sign, file_type="db", prune=False, th=th)
                    pruned_observation_ids = load_file(file, pruned_observation_ids, observation_sign, file_type="db", prune=True, th=th)
                elif filename.startswith("wifi-"):
                    file = osp.join(item, filename)
                    observation_ids = load_file(file, observation_ids, observation_sign, file_type="new", prune=False, th=th)
                    pruned_observation_ids = load_file(file, pruned_observation_ids, observation_sign, file_type="new", prune=True, th=th)

            series_to_file(observation_ids, observation_file)
            series_to_file(pruned_observation_ids, pruned_observation_file)

            ap_ids = generate_ap_ids(ap_file, ap_sign, observation_ids)

            pruned_ap_ids = generate_ap_ids(pruned_ap_file, ap_sign, pruned_observation_ids)

            generate_graph_file(observation_file, ap_file, offset, graph_file)
            generate_graph_file(pruned_observation_file, pruned_ap_file, offset, pruned_graph_file)

            info_dict[raw_data_dir] = {}
            info_dict[raw_data_dir]["observation"] = len(observation_ids)
            info_dict[raw_data_dir]["ap"] = len(ap_ids)
            info_dict[raw_data_dir]["prune observation"] = len(pruned_observation_ids)
            info_dict[raw_data_dir]["prune ap"] = len(pruned_ap_ids)

    print("generation completed.")
    print(info_dict)

    for building_id in building_ids:
        raw_path = osp.abspath(osp.join(os.getcwd(), "./data/output/{}/raw_data").format(building_id))

        obs_path = osp.join(raw_path, "_observations_{}.pkl".format(building_id))
        out_path = osp.join(raw_path, "floors.txt")

        with open(obs_path, 'rb') as f_in1, open(out_path, "w") as f_out:
            series_obs = pickle.load(f_in1)
            series_obs = {series_obs[i][0]:(series_obs[i][1].split(".")[0]).split("_")[-1] for i in range(len(series_obs))}
            f_out.write(str(series_obs))
            print("Site {} done.".format(building_id))