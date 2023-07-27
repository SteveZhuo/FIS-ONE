import argparse
import os
import warnings

import matplotlib
from clustering import *

from order import *

warnings.filterwarnings("ignore")

matplotlib.use('Agg')

def init_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site_id', type=str, default='5cd56c1ce2acfd2d33b6ca5b')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--output_path', type=str, default='./data/output')
    parser.add_argument('--compare_path', type=str, default='./data/compare/mds')
    parser.add_argument('--floor_path', type=str, default='raw_data/floors.txt')
    parser.add_argument('--ap_path', type=str, default='raw_data/_APs_{}.pkl')
    parser.add_argument('--obs_path', type=str, default='raw_data/_observations_{}.pkl')
    parser.add_argument('--emb_path', type=str, default='embedding/dim_{}.txt')
    parser.add_argument('--threshold', type=int, default=100)
    parser.add_argument('--data_description', type=bool, default=False)
    parser.add_argument('--method', type=str, default='SAGE-pairwise-AJ')
    # parser.add_argument('--clustering', type=bool, default=True)
    # parser.add_argument('--order', type=bool, default=True)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--tsne_dim', type=int, default=3)
    parser.add_argument('--k', type=int, default=1) # top k
    parser.add_argument('--order_id', type=int, default=2) # 0: pairwise, 1: jaccard, 2: adapted jaccard
    parser.add_argument('--thresh', type=int, default=-100)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = init_parse()

    for building in os.listdir('./data/input'):
        args.site_id = building
        output_path = osp.abspath(osp.join(os.getcwd(), args.output_path, args.site_id))
        emb_path = osp.abspath(osp.join(os.getcwd(), args.output_path, args.site_id, args.emb_path.format(args.dim)))
        print (emb_path)

        dataset, rows, u_count, new_floor_data = get_embs_with_thresh(args)
        floor_labels = [new_floor_data[i] for i in rows[:u_count]]
        floors = list(set(floor_labels))

        print("Result of site {}".format(args.site_id))
        floor_ids = []
        for item in floors:
            floor_ids.append(floor_map[item])
        indexes = sorted(range(len(floor_ids)), key=lambda k: floor_ids[k])
        floors = [floors[i] for i in indexes]
        print (f"floor ids are {floor_ids}, and indexes are {indexes}")
        print (f"floors are {floors}")
        obs_dataset = dataset[:u_count]
        obs_rows = rows[:u_count]
        ap_dataset = dataset[u_count:]
        ap_rows = rows[u_count:]

        if len(obs_dataset) == 0:
            continue

        if args.data_description:
            print(print_ap_floors(args))

        method_list = args.method.split('-')
        fake_obs_labels = cluster_algo(obs_dataset, n_clusters=len(floors), prefix="F", algo="hierarchy")
        show_performance(fake_obs_labels, floor_labels, building, args)

        print ("Start indexing...")
        predicted_floor_labels_from_ground_truth, occupied_floor_labels = get_predicted_labels(fake_obs_labels, floor_labels)
        floor_ids = []
        for item in occupied_floor_labels:
            floor_ids.append(floor_map[item])
        indexes = sorted(range(len(floor_ids)), key=lambda k: floor_ids[k])
        occupied_floor_labels = [occupied_floor_labels[i] for i in indexes]

        order_2_opt = show_result_2_opt(obs_dataset, occupied_floor_labels,predicted_floor_labels_from_ground_truth, args, building)