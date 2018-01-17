import argparse
import os
import numpy as np
import math

import obj_tools
import neuralnets.grammar as grammar

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle

class ShapeGraph():
   
    def __init__(self, input_list):
        self.node_ids = np.array( [input_list[0::12], input_list[1::12]]).astype(int).T
        self.node_types = np.array( [input_list[2::12], input_list[3::12]] ).astype(int).T
        self.node_unique_types = (np.unique(self.node_types, axis = 0)).tolist()
        self.relative_translations = np.array([input_list[4::12], input_list[5::12], input_list[6::12]]).T
        self.relative_rotations = np.array([input_list[7::12], input_list[8::12], input_list[9::12], input_list[10::12]]).T
        self.sizes = np.array(input_list[11::12])

        #print("node ids:")
        #print(self.node_ids)
        #print("type ids:")
        #print(self.node_types) 
        #print("unique type ids:")
        #print(self.node_unique_types) 

        #print("relative translations: ")
        #print(self.relative_translations)
        #print("relative rotations: ")
        #print(self.relative_rotations)


def get_arguments():
    parser = argparse.ArgumentParser(description="Shape graph configuration estimation from .obj file collections.")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("-o", "--out_plot", type=str,  help="Where to save the edge configuration plot.")
    return parser.parse_args()

def process_folder(folder_name, file_list = []):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            process_folder(subfolfer_name, word_list)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"):
            file_list.append(folder_name + "/" + item_name)

###################################################
# Estimate edge types (categories) via clustering #
###################################################
def categorize_edges(file_list, grammar, out_plot):
    all_node_types = np.empty(dtype=int, shape=(0, 2))
    all_node_ids = np.empty(dtype=int, shape=(0, 2))
    all_relative_translations = np.empty(dtype=float, shape=(0, 3))
    all_sizes = np.empty(dtype=float,  shape=(0, 1))

    
    for file_name in file_list:
        graph = ShapeGraph(obj_tools.obj2graph(file_name))
        if all_node_ids.shape != (0, 2):
            graph.node_ids = np.add(graph.node_ids, 1 + np.amax(all_node_ids))            
        all_node_ids = np.append(all_node_ids, graph.node_ids, axis = 0)
        all_node_types = np.append(all_node_types, graph.node_types, axis = 0)
        all_relative_translations = np.append(all_relative_translations, graph.relative_translations, axis = 0)
        all_sizes = np.append(all_sizes, graph.sizes)

    node_unique_types = (np.unique(all_node_types, axis = 0)).tolist()
    
    fig = plt.figure(figsize=(8.27,11.7 * math.ceil( len(node_unique_types) / 2.0 ) / 3.0) ) #A4 page per 8 subplots
    fig.clf()

    out_cluster_centers = []
    for node_type_pair in node_unique_types:
        ids = np.where((all_node_types == node_type_pair).sum(axis=1) == 2)
        current_node_ids = all_node_ids[ids]
        current_translations = all_relative_translations[ids]
        current_node_size = all_sizes[ids][0]

        if np.prod(ids[0].shape) <= 1:
            out_cluster_centers.append(current_translations)
            continue #should not happen

        # #############################################################################
        # Cluster using KMeans and max node degree as number of clusters
        
        #n_clusters = max([pair[1] for pair in grammar.neighbor_counts if pair[0] == grammar.charset[node_type_pair[0]]])

        #kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        #kmeans.fit(current_translations)
        #labels = kmeans.labels_
        #cluster_centers = kmeans.cluster_centers_          

        # #############################################################################
        # Compute clustering with MeanShift

        #bandwidth = estimate_bandwidth(current_translations, quantile = 0.35)

        #max_extent = np.max(current_translations, axis = 0)
        #min_extent = np.min(current_translations, axis = 0)
        #bbox_diagonal_length = math.sqrt(np.dot(max_extent - min_extent, max_extent - min_extent))
        #bandwidth = max(bbox_diagonal_length * 0.15, 0.01)
        
        bandwidth = 0.2 * current_node_size

        ms = MeanShift(bandwidth = bandwidth, bin_seeding = True)
        ms.fit(current_translations)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_        
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)

        # #############################################################################
        # Compute Affinity Propagation

        #af = AffinityPropagation().fit(current_translations)
        #cluster_centers_indices = af.cluster_centers_indices_
        #labels = af.labels_
        #n_clusters = len(cluster_centers_indices)
        #cluster_centers = current_translations[cluster_centers_indices]

        # #############################################################################
        # Merge spatially close clusters
        unique_ids, counts = np.unique(current_node_ids[:,0], return_counts = True)
        n_clusters_lower_bound = np.amax(counts)
        if n_clusters_lower_bound < n_clusters:
            max_extent = np.max(current_translations, axis = 0)
            min_extent = np.min(current_translations, axis = 0)
            bbox_diagonal_length = np.linalg.norm(max_extent - min_extent)
            
            cluster_merge_candidates = []
            for i in range(n_clusters):
                for j in range(i+1, n_clusters):
                    dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    if  dist < 0.25 * current_node_size:
                        cluster_merge_candidates.append([dist, i, j])
        
            cluster_merge_candidates.sort(key=lambda tup: tup[0])
            cluster_merge_candidates = cluster_merge_candidates[0:n_clusters - n_clusters_lower_bound]
            for candidate_id in range(len(cluster_merge_candidates)):
                _, i, j = cluster_merge_candidates[candidate_id]
                labels[labels == j] = i
                cluster_centers[i] = np.multiply(np.add(cluster_centers[i], cluster_centers[j]), 0.5)
                np.delete(cluster_centers, j, axis = 0)
                labels[labels > j] -= 1
                for k in range(candidate_id + 1, len(cluster_merge_candidates)):
                    if cluster_merge_candidates[k][1] == j:
                        cluster_merge_candidates[k][1]  = i
                    if cluster_merge_candidates[k][1] > j:
                        cluster_merge_candidates[k][1] -= 1
                    if cluster_merge_candidates[k][2] == j:
                        cluster_merge_candidates[k][2]  = i
                    if cluster_merge_candidates[k][2] > j:
                        cluster_merge_candidates[k][2] -= 1

            n_clusters = n_clusters - len(cluster_merge_candidates)  

        # #############################################################################
        # Plot Result
        out_cluster_centers.append(cluster_centers)

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

        subplot_id = 1 + node_unique_types.index(node_type_pair)
        ax = fig.add_subplot((1 + len(node_unique_types)) / 2, 2, subplot_id, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        xyz_max = np.amax(current_translations)
        xyz_min = np.amin(current_translations)
        ax.set_xlim([xyz_min,xyz_max])
        ax.set_ylim([xyz_min,xyz_max])
        ax.set_zlim([xyz_min,xyz_max])

        ax.set_title('edge type [' + grammar.charset[node_type_pair[0]] + ', ' + grammar.charset[node_type_pair[1]] + '] clusters: %d' % n_clusters)
        for k, col in zip(range(n_clusters), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            ax.scatter(current_translations[my_members, 0], current_translations[my_members, 1], current_translations[my_members, 2], c = col, marker = '.')
            ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2], c = col, marker = 'o')

        #for k, col in zip(range(n_clusters), colors):
        #    my_members = labels == k
        #    cluster_center = cluster_centers[k]
        #    plt.plot(current_translations[my_members, 0], current_translations[my_members, 2], col + '.')
        #    plt.plot(cluster_center[0], cluster_center[2], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

        # #############################################################################
        # Compute DBSCAN
        #current_translations = StandardScaler().fit_transform(current_translations)

        #db = DBSCAN(eps=0.3, min_samples=len(file_list)).fit(current_translations)
        #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        #core_samples_mask[db.core_sample_indices_] = True
        #labels = db.labels_

        ## Number of clusters in labels, ignoring noise if present.
        #n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        ## Black removed and is used for noise instead.
        #unique_labels = set(labels)
        #colors = [plt.cm.Spectral(each)
        #          for each in np.linspace(0, 1, len(unique_labels))]
        #for k, col in zip(unique_labels, colors):
        #    if k == -1:
        #        # Black used for noise.
        #        col = [0, 0, 0, 1]

        #    class_member_mask = (labels == k)

        #    xy = current_translations[class_member_mask & core_samples_mask]
        #    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #             markeredgecolor='k', markersize=14)

        #    xy = current_translations[class_member_mask & ~core_samples_mask]
        #    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #             markeredgecolor='k', markersize=6)

        #plt.title('Estimated number of clusters: %d' % n_clusters)
        #plt.show()
        #plt.savefig('plot_' + grammar.charset[node_type_pair[0]] + '_' + grammar.charset[node_type_pair[1]] + '.pdf', bbox_inches='tight')


        print("node types: [" + grammar.charset[node_type_pair[0]] + ", " + grammar.charset[node_type_pair[1]] + "] num clusters : " + str(n_clusters))
    
    if(out_plot != ""):    
        plt.savefig(out_plot, bbox_inches='tight')

    return out_cluster_centers, node_unique_types    


def main():
    args = get_arguments()
    
    file_list = []
    process_folder(args.in_folder, file_list)

    inputA = file_list[0]    
    inputB = file_list[len(file_list) - 1]
    
    initial_smiles_strings = []
    initial_smiles_strings.append(str(obj_tools.obj2string(inputA)))
    initial_smiles_strings.append(str(obj_tools.obj2string(inputB)))
    tile_grammar = grammar.TilingGrammar(initial_smiles_strings)

    out_filename = ""
    if(args.out_plot):
        out_filename = args.out_plot
    
    cluster_centers, node_types = categorize_edges(file_list[:100], tile_grammar, out_filename)
        
    
    #print("cluster centers:")
    #print(cluster_centers)


if __name__ == "__main__":

    main()
