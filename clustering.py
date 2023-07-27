from lib2to3.pgen2.token import VBAR
from termios import FIONBIO
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, OPTICS, SpectralClustering
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score, mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score, calinski_harabasz_score, davies_bouldin_score

file_results_clustering = './data/output/results_clustering_{}.txt'


def cluster_algo(dataset, n_clusters, prefix="obs", algo="hierarchy"):
    if algo == "hierarchy":
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity="euclidean", linkage="ward")
    elif algo == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=0)
    elif algo == "dbscan":
        model = DBSCAN(eps=0.25, min_samples=3)
    elif algo == "optics":
        model = OPTICS(min_samples=5, xi=0.2, p=1)
    elif algo == "spectral":
#         model = SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr') # this mode is for higher sklearn versions
        model = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize')
    else:
        raise ValueError(f"No such algo: {algo}")
    model.fit(dataset)
    return [f"{prefix}{i}" for i in model.labels_]

def check_accuracy(fake_floor_labels, floor_labels): # not used, as we modified floor labelling, see show_performance()
    correct_labels = 0
    counts = 0
    for label in list(set(fake_floor_labels)):
        tmp_floor_labels = [floor_labels[idx] for idx in range(len(fake_floor_labels)) 
                            if fake_floor_labels[idx] == label]
        label_counts = {}
        for item in tmp_floor_labels:
            if item not in label_counts:
                label_counts[item] = 0
            label_counts[item] += 1
        max_count = np.max(list(label_counts.values()))
        correct_labels += max_count
        counts += np.sum(list(label_counts.values()))
    return correct_labels / counts

def evaluate(conf_matrix, building, args=None):
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP

    pred_list = TP+FP
    true_list = TP+FN

    f_list = []
    t_list = []
    for i in range(len(pred_list)):
        for _ in range(int(pred_list[i])):
            f_list.append(i)
        for _ in range(int(true_list[i])):
            t_list.append(i)

    ari = adjusted_rand_score(t_list, f_list)
    nmi = normalized_mutual_info_score(t_list, f_list, average_method='arithmetic')
    print (f'{building}: {len(conf_matrix)}, {ari}, {nmi} \n')
    if args:
        with open(file_results_clustering.format(args.dim), 'a+') as f_out:
            f_out.write(f'{building}: {len(conf_matrix)}, ari: {ari}, nmi: {nmi} \n')

def show_performance(fake_floor_labels, floor_labels, building, args=None): # performance when we "know" the true labels, not used as this is a clustering job
    cluster_dict = {}
    occupied_true_labels = []
    true_floors = list(set(floor_labels))

    for label in list(set(fake_floor_labels)):
        cluster_dict[label] = fake_floor_labels.count(label)
    fake_floor_list = [k for (k, v) in sorted(cluster_dict.items(), key=lambda x:x[1], reverse=True)] # fake floor labels (cluster labels)

    conf_matrix = np.zeros((len(fake_floor_list), len(fake_floor_list))) # newly designed, as we modified the accuracy measure
    for label in (fake_floor_list):
        tmp_floor_labels = [floor_labels[idx] for idx in range(len(fake_floor_labels)) if fake_floor_labels[idx] == label]
        label_counts = {}
        for item in tmp_floor_labels:
            if item not in label_counts:
                label_counts[item] = 0
            label_counts[item] += 1
        for (k, v) in sorted(label_counts.items(), key=lambda x:x[1], reverse=True):
            if k not in occupied_true_labels:
                i = true_floors.index(k)
                occupied_true_labels.append(k)
                conf_matrix[i][i] = v
                break
        for (k, v) in sorted(label_counts.items(), key=lambda x:x[1], reverse=True):
            if k not in occupied_true_labels:
                conf_matrix[true_floors.index(k)][i] = v

    print("Number of clusters: ", len(set(fake_floor_labels)))
    evaluate(conf_matrix, building, args)


def get_predicted_labels(fake_floor_labels, floor_labels):
    cluster_dict = {}
    occupied_true_labels = []
    predicted_floor_labels = ['-1' for item in fake_floor_labels] # store the result after we match fake floor labels to real ones
    true_floors = list(set(floor_labels))

    for label in list(set(fake_floor_labels)):
        cluster_dict[label] = fake_floor_labels.count(label)
    fake_floor_list = [k for (k, v) in sorted(cluster_dict.items(), key=lambda x: x[1],
                                              reverse=True)]  # fake floor labels (cluster labels)

    conf_matrix = np.zeros(
        (len(fake_floor_list), len(fake_floor_list)))  # newly designed, as we modified the accuracy measure
    labeled = []
    for label in (fake_floor_list):
        tmp_floor_labels = [floor_labels[idx] for idx in range(len(fake_floor_labels)) if
                            fake_floor_labels[idx] == label]
        label_counts = {} # Find the majority of true labels in the clusters.
        for item in tmp_floor_labels:
            if item not in label_counts:
                label_counts[item] = 0
            label_counts[item] += 1
        for (k, v) in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            if k not in occupied_true_labels:
                labeled.append(label)
                i = true_floors.index(k)
                occupied_true_labels.append(k)
                conf_matrix[i][i] = v
                indexes = [idx for idx in range(len(fake_floor_labels)) if
                            fake_floor_labels[idx] == label]
                for idx in indexes:
                    predicted_floor_labels[idx] = k
                break

    for label in fake_floor_list:
        if label not in labeled:
            indexes = [idx for idx in range(len(fake_floor_labels)) if
                       fake_floor_labels[idx] == label]
            for true_label in true_floors:
                if true_label not in occupied_true_labels:
                    occupied_true_labels.append(true_label)
                    for idx in indexes:
                        predicted_floor_labels[idx] = true_label
                    break

    return predicted_floor_labels, occupied_true_labels