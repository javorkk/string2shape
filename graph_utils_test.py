import argparse
import os
import numpy as np
import math

import obj_tools
import neuralnets.grammar as grammar

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation

import matplotlib.pyplot as plt
from itertools import cycle

class ShapeGraph():
   
    def __init__(self, input_list):
        self.node_ids = np.array( [input_list[0::11], input_list[1::11]]).astype(int).T
        self.node_types = np.array( [input_list[2::11], input_list[3::11]] ).astype(int).T
        self.node_unique_types = (np.unique(self.node_types, axis = 0)).tolist()
        self.relative_translations = np.array([input_list[4::11], input_list[5::11], input_list[6::11]]).T
        self.relative_rotations = np.array([input_list[7::11], input_list[8::11], input_list[9::11], input_list[10::11]]).T

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
    parser = argparse.ArgumentParser(description="SMILES string to Wavefront .obj conversion by file search.")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
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
def categorize_edges(file_list, grammar):
    all_node_types = np.empty(dtype=int, shape=[0, 2])
    all_relative_translations = np.empty(dtype=float, shape=[0, 3])
    
    for file_name in file_list:
        graph = ShapeGraph(obj_tools.obj2graph(file_name))
        all_node_types = np.append(all_node_types, graph.node_types, axis = 0)
        all_relative_translations = np.append(all_relative_translations, graph.relative_translations, axis = 0)

    node_unique_types = (np.unique(all_node_types, axis = 0)).tolist()

    for node_type_pair in node_unique_types:
        ids = np.where((all_node_types == node_type_pair).sum(axis=1) == 2)
        current_translations = all_relative_translations[ids]

        # #############################################################################
        # Cluster using KMeans and max node degree as number of clusters
        nbr_count = grammar.neighbor_counts[node_type_pair[0] - 1] # -1 because the first element of neigborcounts is empty
        n_clusters = nbr_count[1]
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        kmeans.fit(current_translations)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        
        # #############################################################################
        # Compute clustering with MeanShift

        #bandwidth = estimate_bandwidth(current_translations, quantile = 0.15)

        #max_extent = np.max(current_translations, axis = 0)
        #min_extent = np.min(current_translations, axis = 0)
        #bbox_diagonal_length = math.sqrt(np.dot(max_extent - min_extent, max_extent - min_extent))
        #bandwidth = max(bbox_diagonal_length * 0.25, 0.01)

        #ms = MeanShift(bandwidth = bandwidth, bin_seeding = True)
        #ms.fit(current_translations)
        #labels = ms.labels_
        #cluster_centers = ms.cluster_centers_        
        #labels_unique = np.unique(labels)
        #n_clusters = len(labels_unique)

        # #############################################################################
        # Compute Affinity Propagation

        #af = AffinityPropagation().fit(current_translations)
        #cluster_centers_indices = af.cluster_centers_indices_
        #labels = af.labels_
        #n_clusters = len(cluster_centers_indices)
        #cluster_centers = current_translations[cluster_centers_indices]

        print("node types:")
        print(node_type_pair)
        print("num clusters: " + str(n_clusters))
    
        plt.figure(1)
        plt.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(current_translations[my_members, 0], current_translations[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        plt.title('Estimated number of clusters: %d' % n_clusters)
        plt.savefig('plot_' + str(node_type_pair[0]) + '_' + str(node_type_pair[1]) + '.pdf', bbox_inches='tight')
            


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

    categorize_edges(file_list, tile_grammar)


if __name__ == "__main__":

    main()
