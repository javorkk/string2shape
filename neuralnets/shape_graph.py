from __future__ import print_function #pylint bug workaround
import math
import numpy as np
import h5py
import random

import obj_tools
import neuralnets.grammar as grammar

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle

class ShapeGraph():

    def __init__(self, input_list):
        self.node_ids = np.array([input_list[0::16], input_list[1::16]]).astype(int).T
        self.node_types = np.array([input_list[2::16], input_list[3::16]] ).astype(int).T
        self.node_unique_types = (np.unique(self.node_types, axis=0)).tolist()
        self.relative_translations = np.array([input_list[4::16], input_list[5::16], input_list[6::16]]).T
        self.relative_rotations = np.array([input_list[7::16], input_list[8::16], input_list[9::16], input_list[10::16]]).T
        self.sizes = np.array(input_list[11::16])
        self.abs_rotations = np.array([input_list[12::16], input_list[13::16], input_list[14::16], input_list[15::16]]).T

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

    def load(self, filename):
        h5f = h5py.File(filename, "r")
        self.node_ids = h5f["node_ids"][:]
        self.node_types = h5f["node_types"][:]
        self.node_unique_types = h5f["node_unique_types"][:].tolist()
        self.relative_translations = h5f["relative_translations"][:]
        self.relative_rotations = h5f["relative_rotations"][:]
        self.sizes = h5f["sizes"][:]
        self.abs_rotations = h5f["abs_rotations"][:]
        h5f.close()

    def store(self, filename):
        h5f = h5py.File(filename, "w")
        h5f.create_dataset("node_ids", data=self.node_ids)
        h5f.create_dataset("node_types", data=self.node_types)
        h5f.create_dataset("node_unique_types", data=self.node_unique_types)
        h5f.create_dataset("relative_translations", data=self.relative_translations)
        h5f.create_dataset("relative_rotations", data=self.relative_rotations)
        h5f.create_dataset("sizes", data=self.sizes)
        h5f.create_dataset("abs_rotations", data=self.abs_rotations)
        h5f.close()


###################################################
# Estimate edge types (categories) via clustering #
###################################################
def categorize_edges(file_list, t_grammar, out_plot = None):
    all_node_types = np.empty(dtype=int, shape=(0, 2))
    all_node_ids = np.empty(dtype=int, shape=(0, 2))
    all_relative_translations = np.empty(dtype=float, shape=(0, 3))
    all_sizes = np.empty(dtype=float, shape=(0, 1))

    for file_name in file_list:
        graph = ShapeGraph(obj_tools.obj2graph(file_name))
        if all_node_ids.shape != (0, 2):
            graph.node_ids = np.add(graph.node_ids, 1 + np.amax(all_node_ids))
        all_node_ids = np.append(all_node_ids, graph.node_ids, axis=0)
        all_node_types = np.append(all_node_types, graph.node_types, axis=0)
        all_relative_translations = np.append(all_relative_translations, graph.relative_translations, axis = 0)
        all_sizes = np.append(all_sizes, graph.sizes)

    node_unique_types = (np.unique(all_node_types, axis=0)).tolist()
    #A4 page per 8 subplots
    fig = plt.figure(figsize=(8.27, 11.7 * math.ceil(len(node_unique_types)/ 2.0)/ 3.0))
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

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
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
        unique_ids, counts = np.unique(current_node_ids[:,0], return_counts=True)
        n_clusters_lower_bound = np.amax(counts)
        if n_clusters_lower_bound < n_clusters:
            max_extent = np.max(current_translations, axis=0)
            min_extent = np.min(current_translations, axis=0)
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
                np.delete(cluster_centers, j, axis=0)
                labels[labels > j] -= 1
                for k in range(candidate_id + 1, len(cluster_merge_candidates)):
                    if cluster_merge_candidates[k][1] == j:
                        cluster_merge_candidates[k][1] = i
                    if cluster_merge_candidates[k][1] > j:
                        cluster_merge_candidates[k][1] -= 1
                    if cluster_merge_candidates[k][2] == j:
                        cluster_merge_candidates[k][2] = i
                    if cluster_merge_candidates[k][2] > j:
                        cluster_merge_candidates[k][2] -= 1

            n_clusters = n_clusters - len(cluster_merge_candidates)

        out_cluster_centers.append(cluster_centers)
        # #############################################################################
        # Plot result_str

        if(out_plot != None):
            colors = cycle(['#7e1e9c','#0343df','#653700','#95d0fc','#029386','#f97306','#96f97b','#c20078','#ffff14','#89fe05','#9a0eea','#06c2ac','#0c06f7','#61de2a','#3778bf','#2242c7','#533cc6','#9bb53c','#05ffa6','#1f6357','#017374','#0cb577','#0a5f38','#ff0789','#afa88b','#08787f','#dd85d7','#a6c875','#a7ffb5','#c2b709','#e78ea5','#966ebd','#ccad60','#ac86a8','#947e94','#983fb2','#ff63e9','#b2fba5','#63b365','#8ee53f','#b7e1a1','#ff6f52','#bdf8a3','#d3b683','#fffcc4','#430541','#ffb2d0','#997570','#ad900d','#c48efd','#507b9c','#7d7103','#fffd78','#da467d','#410200','#c9d179','#fffa86','#5684ae','#6b7c85','#6f6c0a','#7e4071','#009337','#d0e429','#fff917','#1d5dec','#054907','#b5ce08','#8fb67b','#c8ffb0','#fdde6c','#ffdf22','#a9be70','#6832e3','#fdb147','#c7ac7d','#fff39a','#850e04','#efc0fe','#40fd14','#b6c406'])

            subplot_id = 1 + node_unique_types.index(node_type_pair)
            ax = fig.add_subplot((1 + len(node_unique_types)) / 2, 2, subplot_id, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            xyz_max = np.amax(current_translations)
            xyz_min = np.amin(current_translations)
            ax.set_xlim([xyz_min, xyz_max])
            ax.set_ylim([xyz_min, xyz_max])
            ax.set_zlim([xyz_min, xyz_max])

            ax.set_title('edge type [' + t_grammar.charset[node_type_pair[0]] + ', ' + t_grammar.charset[node_type_pair[1]] + '] clusters: %d' % n_clusters)
            for k, col in zip(range(n_clusters), colors):
                my_members = labels == k
                cluster_center = cluster_centers[k]
                ax.scatter(current_translations[my_members, 0], current_translations[my_members, 1], current_translations[my_members, 2], c=col, marker='.')
                ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2], c=col, marker='o')

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
        #plt.savefig('plot_' + t_grammar.charset[node_type_pair[0]] + '_' + t_grammar.charset[node_type_pair[1]] + '.pdf', bbox_inches='tight')


        print("node types: [" + t_grammar.charset[node_type_pair[0]] + ", " + t_grammar.charset[node_type_pair[1]] + "] num clusters : " + str(n_clusters))

    if out_plot != None:
        plt.savefig(out_plot, bbox_inches='tight')

    return out_cluster_centers, node_unique_types

###################################################
# Compute a category sequence for a SMILES string #
###################################################
def smiles_to_edge_categories(word, node_ids, cluster_centers, graph, t_grammar):
    dummy_node_id = len(node_ids)

    num_nodes = 0
    padded_node_ids = []
    for char_id, _ in enumerate(word):
        if word[char_id] in t_grammar.charset:
            padded_node_ids.append(node_ids[num_nodes])
            num_nodes += 1
        else:
            padded_node_ids.append(dummy_node_id)
    padded_node_ids.append(dummy_node_id) #ensure at least one occurence

    edge_list = t_grammar.smiles_to_edges(word, padded_node_ids)

    num_categories = 0
    categories_prefix = [0]
    for clusters in cluster_centers:
        num_categories += clusters.shape[0]
        categories_prefix.append(num_categories)

    edge_categories = []
    for node_id_pair in edge_list:
        if node_id_pair == [dummy_node_id, dummy_node_id]:
            edge_categories.append(num_categories)
        else:
            edge_index = np.where((np.array(node_id_pair) == graph.node_ids).sum(axis=1) == 2)[0]
            # type_id_pair = graph.node_types[edge_index]
            # if type_id_pair.shape[0] == 0:
            #     #work around missing edges in graph.node_ids
            #     char_a = word[padded_node_ids.index(node_id_pair[0])]
            #     char_b = word[padded_node_ids.index(node_id_pair[1])]
            #     type_id_pair = np.array([t_grammar.charset.index(char_a), t_grammar.charset.index(char_b)])
            
            # cluster_set_id = graph.node_unique_types.index(type_id_pair.reshape(2).tolist())
            
            char_a = word[padded_node_ids.index(node_id_pair[0])]
            char_b = word[padded_node_ids.index(node_id_pair[1])]
            type_pair = [char_a, char_b]
            cluster_set_id = t_grammar.neighbor_types.index(type_pair)
            
            relative_translation = graph.relative_translations[edge_index]
            closest_cluster_center_id = num_categories
            dist = float("inf")
            for i in range(cluster_centers[cluster_set_id].shape[0]):
                current = np.linalg.norm(cluster_centers[cluster_set_id][i] - relative_translation)
                if current < dist:
                    dist = current
                    closest_cluster_center_id = i
            edge_categories.append(closest_cluster_center_id + categories_prefix[cluster_set_id])

    return edge_categories

def smiles_to_all_edge_categories(word, node_ids, cluster_centers, graph, t_grammar):
    dummy_node_id = len(node_ids)

    num_nodes = 0
    padded_node_ids = []
    for char_id, _ in enumerate(word):
        if word[char_id] in t_grammar.charset:
            padded_node_ids.append(node_ids[num_nodes])
            num_nodes += 1
        else:
            padded_node_ids.append(dummy_node_id)
    padded_node_ids.append(dummy_node_id) #ensure at least one occurence

    initial_edge_list = t_grammar.smiles_to_edges(word, padded_node_ids)
    edge_list = []
    for edge in initial_edge_list:
        edge_list.append(edge)
        edge_list.append([edge[1], edge[0]])

    num_categories = 0
    categories_prefix = [0]
    for clusters in cluster_centers:
        num_categories += clusters.shape[0]
        categories_prefix.append(num_categories)

    edge_categories = []
    output_edges = []
    for node_id_pair in edge_list:
        if node_id_pair == [dummy_node_id, dummy_node_id]:
            continue
            #edge_categories.append(num_categories)
        else:
            edge_index = np.where((np.array(node_id_pair) == graph.node_ids).sum(axis=1) == 2)[0]
            char_a = word[padded_node_ids.index(node_id_pair[0])]
            char_b = word[padded_node_ids.index(node_id_pair[1])]
            type_pair = [char_a, char_b]
            cluster_set_id = t_grammar.neighbor_types.index(type_pair)
            
            relative_translation = graph.relative_translations[edge_index]
            closest_cluster_center_id = num_categories
            dist = float("inf")
            for i in range(cluster_centers[cluster_set_id].shape[0]):
                current = np.linalg.norm(cluster_centers[cluster_set_id][i] - relative_translation)
                if current < dist:
                    dist = current
                    closest_cluster_center_id = i
            edge_categories.append(closest_cluster_center_id + categories_prefix[cluster_set_id])
            output_edges.append(node_id_pair)

    return edge_categories, output_edges

#TODO implement
#def smiles_to_bidir_edge_categories(word, node_ids, cluster_centers, graph, t_grammar):

def smiles_substring(node_id, visited, adjacency_lists, cycle_ids, node_types, t_grammar):
    visited[node_id] = 1
    neighbor_list = adjacency_lists[node_id]
    random.shuffle(neighbor_list)

    dummy_node_id = len(visited)

    result_str = t_grammar.charset[node_types[node_id]]
    result_node_list = [node_id]
    
    cycle_labels = ""
    for cycle_id in cycle_ids[node_id]:
        if cycle_labels == "":
            cycle_labels += str(cycle_id)
        else:
            cycle_labels += t_grammar.NUM_DELIMITER + str(cycle_id)
    result_str += cycle_labels
    result_node_list += [dummy_node_id for ch in cycle_labels]
    
    last_subtree = ""
    last_nodes = []
    other_subtrees = ""
    other_nodes = []

    for nbr_id in neighbor_list:
        if visited[nbr_id]:
            continue
        subtree_str, subtree_nodes = smiles_substring(nbr_id, visited, adjacency_lists, cycle_ids, node_types, t_grammar)
        if last_subtree == "":
            last_subtree = subtree_str
            last_nodes = subtree_nodes
        else:
            other_subtrees += t_grammar.BRANCH_START
            other_subtrees += subtree_str
            other_subtrees += t_grammar.BRANCH_END
            
            other_nodes += [dummy_node_id]
            other_nodes += subtree_nodes
            other_nodes += [dummy_node_id]


    result_str += other_subtrees
    result_str += last_subtree

    result_node_list += other_nodes
    result_node_list += last_nodes

    return result_str, result_node_list

def smiles_variations(word, padded_node_ids, t_grammar, num_variations=10):
   
    num_nodes = 0
    dummy_node_id = max(padded_node_ids)
    node_ids = [idx for idx in padded_node_ids if idx != dummy_node_id]
    node_types = [len(t_grammar.charset) for i in node_ids]    

    for char_id, _ in enumerate(word):
        if word[char_id] in t_grammar.charset:
            node_types[node_ids[num_nodes]] = t_grammar.charset.index(word[char_id])
            num_nodes += 1

    edge_list = t_grammar.smiles_to_edges(word, padded_node_ids)
    
    adjacency_lists = []
    cycle_lists = []
    for node_id in range(num_nodes):
        current_neighbors = set()
        current_cycles = set()
        for idx, edge in enumerate(edge_list):            
            if edge[0] == node_id:
                if word[idx] in t_grammar.charset:
                    current_neighbors.add(edge[1])
                elif word[idx] in t_grammar.DIGITS:
                    current_cycles.add(edge[1])
            if edge[1] == node_id:
                if word[idx] in t_grammar.charset:
                    current_neighbors.add(edge[0])
                elif word[idx] in t_grammar.DIGITS:
                    current_cycles.add(edge[0])
        adjacency_lists.append(sorted(list(current_neighbors)))
        cycle_lists.append(sorted(list(current_cycles)))
    
    cycle_ids = []
    cycle_count = 0
    for node_id in range(num_nodes):
        local_ids = []
        for cyclic_node_id in cycle_lists[node_id]:
            if cyclic_node_id > node_id:
                local_ids.append(cycle_count)
                cycle_count += 1
            else:
                local_ids.append(cycle_ids[cyclic_node_id][cycle_lists[cyclic_node_id].index(node_id)])                                      
        cycle_ids.append(local_ids)

    result_str = ""
    result_node_lists = []
    for iteration_id in range(num_variations):
        visited = []
        for node_id in range(num_nodes):
            visited.append(0)
        start_node_id = random.randint(0, num_nodes - 1)
        new_string, node_list = smiles_substring(start_node_id, visited, adjacency_lists, cycle_ids, node_types, t_grammar)
        #if t_grammar.check_word(new_string):
        result_str += new_string
        result_node_lists.append(node_list)
        if iteration_id < num_variations - 1:
            result_str += "\n"
    
    return result_str.split("\n"), result_node_lists
