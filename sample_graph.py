from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys
import random
import networkx as nx
import obj_tools

from neuralnets.autoencoder import TilingVAE, Tiling_LSTM_VAE, Tiling_LSTM_VAE_
from neuralnets.utils import one_hot_array, one_hot_index, from_one_hot_array, decode_smiles_from_indexes
from neuralnets.utils import load_dataset
import neuralnets.grammar as grammar

LATENT_DIM = 292
TARGET = 'autoencoder'
NUM_SAMPLES = 1000
GRAPH_SIZE = 10000
GRAPH_K = 4
MODEL_TYPE = 'simple'

TREE_GRAMMAR = True

def get_arguments():
    parser = argparse.ArgumentParser(description='Shape sampling network')
    parser.add_argument('input_data', type=str, help='Input sample set.')
    parser.add_argument('model', type=str, help='Trained Keras model to use.')
    parser.add_argument('grammar', type=str, help='Tiling grammar.')
    parser.add_argument('latent_data', type=str, help='File of latent representation tensors for decoding.')
    parser.add_argument('latent_graph', type=str, help='File of latent graph for sampling.')
    parser.add_argument('--folder_name', type=str, default="", help='Where to search for pre-built examples.')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES, help='Number of sample paths for data augmentation.')
    parser.add_argument('--graph_size', type=int, default=GRAPH_SIZE, help='Size of latent graph.')
    parser.add_argument('--graph_degree', type=int, default=GRAPH_K, help='Minimum node degree.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM, help='Dimensionality of the latent representation.')
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE, help='What type model to train: simple, lstm.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output.')
    return parser.parse_args()

def str_to_file(folder_name, query_word, tiling_grammar):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            str_to_file(subfolfer_name, query_word, tiling_grammar)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"): 
            #current_str = obj_tools.obj2string(folder_name + "/" + item_name)
            current_strings = obj_tools.obj2strings(folder_name + "/" + item_name).split("\n")

            for current_str in current_strings:
                mismatch = False
                if TREE_GRAMMAR == False:
                    for i in range(len(tiling_grammar.DIGITS)):
                        if(query_word.count(tiling_grammar.DIGITS[i]) != current_str.count(tiling_grammar.DIGITS[i])):
                            mismatch = True
                            break#different number of cycles
                for i in range(1, len(tiling_grammar.charset)):
                    if(query_word.count(tiling_grammar.charset[i]) != current_str.count(tiling_grammar.charset[i])):
                        mismatch = True
                        break
                #if tiling_grammar.similar_words(query_word, current_str):
                if not mismatch:
                    return True, item_name
    return False, ""

def read_latent_data(filename):
    h5f = h5py.File(filename, 'r')
    data = h5f['latent_vectors'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    return (data, charset)

def _gen_latent_path(data, end_pt_0, end_pt_1, waypoints = [], depth = 0):
    if depth > 4:
        return waypoints
    
    #sample_ids = np.random.randint(0, len(data), 512)
    #sample_dist = [np.linalg.norm(data[end_pt_0] - data[x]) + np.linalg.norm(data[end_pt_1] - data[x]) for x in sample_ids]
    sample_dist = [np.linalg.norm(data[end_pt_0] - data[x]) + np.linalg.norm(data[end_pt_1] - data[x]) for x in range(len(data)) if x != end_pt_1 and x != end_pt_0]

    val, idx = min((val, idx) for (idx, val) in enumerate(sample_dist))
    _gen_latent_path(data, end_pt_0, idx, waypoints, depth + 1)
    waypoints.append(idx)
    _gen_latent_path(data, idx, end_pt_1, waypoints, depth + 1)

    return waypoints

def load_input(args):
    if not os.path.isfile(args.input_data):
        raise ValueError("Input file %s doesn't exist" % args.input_data)

    data_train, data_test, charset= load_dataset(args.input_data)

    model = TilingVAE()
    if args.model_type == 'lstm':
        model = Tiling_LSTM_VAE()
    elif args.model_type == 'lstm_':
        model = Tiling_LSTM_VAE_()

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = args.latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    tiling_grammar = grammar.TilingGrammar([])
    if os.path.isfile(args.grammar):
        tiling_grammar.load(args.grammar)
    else:
        raise ValueError("Grammar file %s doesn't exist" % args.grammar)
    
    if TREE_GRAMMAR:
        tiling_grammar.convert_to_tree_grammar()

    data = np.append(data_train, data_test, axis=0)
    latent_data = model.encoder.predict(data)

    return model, tiling_grammar, latent_data, charset

def gen_latent_points(args):
    latent_dim = args.latent_dim
    model, tiling_grammar, latent_data, charset = load_input(args)

    initial_size = latent_data.shape[0]

    for i in range(args.num_samples):
        sample_ids = np.random.randint(0, latent_data.shape[0], 2)

        decoded_data_0 = model.decoder.predict(latent_data[sample_ids[0]].reshape(1, latent_dim)).argmax(axis=2)[0]
        char_data_0 = decode_smiles_from_indexes(decoded_data_0, charset)

        decoded_data_1 = model.decoder.predict(latent_data[sample_ids[1]].reshape(1, latent_dim)).argmax(axis=2)[0]
        char_data_1 = decode_smiles_from_indexes(decoded_data_1, charset)

        if not (tiling_grammar.check_word(char_data_0) and tiling_grammar.check_word(char_data_1)):
            if args.verbose:
                print("---------------------invalid decoded word---------------------")
                if not tiling_grammar.check_word(char_data_0, verbose=True):
                    print("(word 1)")
                if not tiling_grammar.check_word(char_data_1, verbose=True):
                    print("(word 2)")
                print("--------------------------------------------------------------")
            continue

        if args.verbose:
            print("---------------------sample " + str(i) + "------------------------------------------")
            print("data point  0.0: " + char_data_0)

        path_ids = []
        path_ids.append(sample_ids[0])
        path_ids = _gen_latent_path(latent_data, sample_ids[0], sample_ids[1], waypoints = path_ids)
        path_ids.append(sample_ids[1])

        for p in range(len(path_ids) - 1):                   
            decoded_data_p = model.decoder.predict(latent_data[path_ids[p + 1]].reshape(1, latent_dim)).argmax(axis=2)[0]
            char_data_p = decode_smiles_from_indexes(decoded_data_p, charset)
            if not tiling_grammar.check_word(char_data_p):
                continue

            for k in [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0]:
                current_distance = np.linalg.norm(latent_data[path_ids[p]] - latent_data[path_ids[p + 1]])
                rnd_offset = np.array([np.random.random(latent_dim)]) * 0.1 * current_distance
                z_sample = (1.0 - k) * latent_data[path_ids[p]] + k * latent_data[path_ids[p + 1]] + rnd_offset
                decoded_sample_k = model.decoder.predict(z_sample.reshape(1, latent_dim)).argmax(axis=2)[0]
                char_sample_k = decode_smiles_from_indexes(decoded_sample_k, charset)
                if not tiling_grammar.check_word(char_sample_k):
                    continue
                if tiling_grammar.similar_words(char_sample_k, char_data_p):
                    continue
                if tiling_grammar.similar_words(char_sample_k, char_data_0):
                    continue
                if tiling_grammar.similar_words(char_sample_k, char_data_1):
                    continue
                latent_data =  np.append(latent_data, z_sample, axis=0)
                if args.verbose:
                    print("sample point " + str(k) + ": "  + char_sample_k + " (rnd offset = " + str(0.1 * current_distance) + ")")
                break

            if args.verbose and p < len(path_ids) - 2 :
                print("path waypoint " + str(p + 1) + ": "  + char_data_p)

        if args.verbose:
            print("data point   1.0: " + char_data_1)
            print("-----------------------------------------------------------------------")

    print("Discovered ", len(latent_data) - initial_size, " additional valid points in latent space.")

    if not os.path.isfile(args.latent_data):
        h5f = h5py.File(args.latent_data, 'w')
        h5f.create_dataset('charset', data=charset)
        h5f.create_dataset('latent_vectors', data=latent_data)
        h5f.close()
    else:
        print("Latent data already exists! Rename or remove the file.")

def build_latent_graph(args):

    if args.graph_degree >= args.graph_size:
        raise ValueError("Requested graph degree %s larger than graph size %s" % (args.graph_degree, args.model))

    if not os.path.isfile(args.input_data):
        raise ValueError("Input file %s doesn't exist" % args.input_data)

    model, tiling_grammar, latent_data, charset = load_input(args)

    permuted_ids = np.random.permutation(len(latent_data))
    selected_ids = []
    words = []

    for idx in permuted_ids:
        decoded_data = model.decoder.predict(latent_data[idx].reshape(1, args.latent_dim)).argmax(axis=2)[0]
        word = decode_smiles_from_indexes(decoded_data, charset)

        if not tiling_grammar.check_word(word):
            continue
        selected_ids.append(idx)
        words.append(word)
        if len(selected_ids) >= args.graph_size:
            break

    search_graph = nx.Graph()
    #graph nodes
    search_graph.add_nodes_from(selected_ids)
    #graph edges
    for i, idx in enumerate(selected_ids):
        #add an edge to each similar word
        for j, idy in enumerate(selected_ids):
            if tiling_grammar.similar_words(words[i], words[j]):
                search_graph.add_edge(idx, idy, weight=0.0)

        #connect to k-nearest points in latent space
        dist_id_pairs = []
        for j in range(len(selected_ids)):
            dist = np.linalg.norm(latent_data[selected_ids[i]] - latent_data[selected_ids[j]])
            dist_id_pairs.append((dist, j))
            if len(dist_id_pairs) % args.graph_degree == 0:
                dist_id_pairs = sorted(dist_id_pairs)
                dist_id_pairs = dist_id_pairs[:args.graph_degree]

        dist_id_pairs = sorted(dist_id_pairs)
        dist_id_pairs = dist_id_pairs[:args.graph_degree]

        for d, j in dist_id_pairs:
            similarity = tiling_grammar.word_similarity(words[i], words[j])
            idy = selected_ids[j]
            search_graph.add_edge(idx, idy, weight=similarity)

    nx.write_graphml(search_graph, args.latent_graph)

def sample_path(args):

    latent_data, charset = read_latent_data(args.latent_data)

    model, tiling_grammar, latent_data, charset = load_input(args)

    if not os.path.isfile(args.latent_graph):
        raise ValueError("Search graph file %s doesn't exist" % args.latent_graph)

    search_graph = nx.read_graphml(args.latent_graph)
    node_list = [int(x) for x in list(search_graph.nodes)]

    for i in range(args.num_samples):
        samples = np.random.randint(0, len(node_list), 2)
        sample_ids = [node_list[samples[0]], node_list[samples[1]]]

        decoded_data_0 = model.decoder.predict(latent_data[sample_ids[0]].reshape(1, args.latent_dim)).argmax(axis=2)[0]
        char_data_0 = decode_smiles_from_indexes(decoded_data_0, charset)

        decoded_data_1 = model.decoder.predict(latent_data[sample_ids[1]].reshape(1, args.latent_dim)).argmax(axis=2)[0]
        char_data_1 = decode_smiles_from_indexes(decoded_data_1, charset)
        if not (tiling_grammar.check_word(char_data_0) and tiling_grammar.check_word(char_data_1)) :
            continue

        file_name_0 = "?"
        file_name_1 = "?"
        if args.folder_name != "":
            found0, file_name_0 = str_to_file(args.folder_name, char_data_0, tiling_grammar)
            if not found0:
                continue
            found1, file_name_1 = str_to_file(args.folder_name, char_data_1, tiling_grammar)
            if not found1:
                continue

        shortest_path = nx.shortest_path(search_graph, source=str(sample_ids[0]), target=str(sample_ids[1]), weight='weight')

        if len(shortest_path) < 5:
            continue

        decoded_words = [char_data_0]
        valid_words = [True]

        for pt_id in shortest_path[1:-1]:
            for j in range(64):
                decoded_data = model.decoder.predict(latent_data[int(pt_id)].reshape(1, args.latent_dim)).argmax(axis=2)[0]
                word =  decode_smiles_from_indexes(decoded_data, charset)
                if tiling_grammar.check_word(word):
                    duplicate = tiling_grammar.similar_words(word, char_data_1)
                    for previous, valid_flag in zip(decoded_words, valid_words):
                        if not valid_flag:
                            continue
                        if tiling_grammar.similar_words(word, previous):
                            duplicate = True
                    if not duplicate:
                        decoded_words.append(word)
                        valid_words.append(True)
                        break
                elif j == 63:
                    decoded_words.append(word)
                    valid_words.append(False)

        decoded_words.append(char_data_1)
        valid_words.append(True)

        if valid_words.count(True) < 5:
            continue

        print("---------------------path sample " + str(i) + "------------------------------------------")
        print("start  :", decoded_words[0]," file: ", file_name_0)
        file_name_w = "?"
        for w, flag in zip(decoded_words, valid_words)[1:-1]:
            if flag:
                if args.folder_name != "":
                    _, file_name_w = str_to_file(args.folder_name, w, tiling_grammar)
                print("valid  :", w, " file: ", file_name_w)
            else:
                print("invalid:", w)
        print("end    :", decoded_words[-1], " file: ", file_name_1)
        print("----------------------------------------------------------------------------------")

def main():
    args = get_arguments()

    if not os.path.isfile(args.latent_data):
        print("Creating latent sample locations...")
        gen_latent_points(args)

    if not os.path.isfile(args.latent_graph):
        print("Generating graph in latent space...")
        build_latent_graph(args)

    sample_path(args)

if __name__ == '__main__':
    main()
