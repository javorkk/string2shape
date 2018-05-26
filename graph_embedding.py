from __future__ import print_function
import argparse
import os
import numpy as np

import obj_tools
import neuralnets.shape_graph as shape_graph

from neuralnets.seq2seq import Seq2SeqAE, Seq2SeqRNN, Seq2SeqNoMaskRNN
from neuralnets.grammar import TilingGrammar
from neuralnets.utils import load_categories_dataset, decode_smiles_from_indexes, from_one_hot_brray
from neuralnets.shape_graph import ShapeGraph, smiles_variations, categorize_edges
from train_seq2seq import decode_sequence

def get_arguments():
    parser = argparse.ArgumentParser(description='Sequence to sequence autoencoder network')
    parser.add_argument('in_folder', type=str, help='Folder with example objects.')
    parser.add_argument('model', type=str, help='The trained seq2seq model.')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('grammar', type=str, help='The HDF5 file with the tiling grammar.')
    parser.add_argument('out', type=str, help='Where to save the output file. If this file exists, it will be overwritten.')
    return parser.parse_args()

def process_folder(folder_name, file_list = []):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            process_folder(subfolfer_name, file_list)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"):
            file_list.append(folder_name + "/" + item_name)

def embed_sequence(s2s_model, grammar, input_charset, input_word, example_graph, example_categories):
    return 0

def main():
    args = get_arguments()
    
    file_list = []
    process_folder(args.in_folder, file_list)

    inputA = file_list[0]    
    inputB = file_list[len(file_list) - 1]

    initial_smiles_strings = []
    initial_smiles_strings.append(str(obj_tools.obj2string(inputA)))
    initial_smiles_strings.append(str(obj_tools.obj2string(inputB)))
    tile_grammar = TilingGrammar([])
    tile_grammar.load(args.grammar)

    cluster_centers, node_types = shape_graph.categorize_edges(file_list[:10], tile_grammar, args.out_plot)

    str_node_ids_a = str(obj_tools.obj2strings_ids(inputA))
    str_node_ids_list_a = str_node_ids_a.split("\n")
    smiles_strings_a = str_node_ids_list_a[:len(str_node_ids_list_a) / 2]
    node_ids_list_a = str_node_ids_list_a[len(str_node_ids_list_a) / 2:]
    
    node_ids_a = []
    for node_list in node_ids_list_a:
        node_ids_a.append([int(i) for i in node_list.split(" ")])

    graph_edges_a = shape_graph.ShapeGraph(obj_tools.obj2graph(inputA))
    edge_categories_a = shape_graph.smiles_to_edge_categories(smiles_strings_a[0], node_ids_a[0], cluster_centers, graph_edges_a,  tile_grammar)


    str_node_ids_b = str(obj_tools.obj2strings_ids(inputB))
    str_node_ids_list_b = str_node_ids_b.split("\n")
    smiles_strings_b = str_node_ids_list_b[:len(str_node_ids_list_b) / 2]
    node_ids_list_b = str_node_ids_list_b[len(str_node_ids_list_b) / 2:]
    
    node_ids_b = []
    for node_list in node_ids_list_b:
        node_ids_b.append([int(i) for i in node_list.split(" ")])

    graph_edges_b = shape_graph.ShapeGraph(obj_tools.obj2graph(inputB))
    edge_categories_b = shape_graph.smiles_to_edge_categories(smiles_strings_b[0], node_ids_b[0], cluster_centers, graph_edges_b,  tile_grammar)


    data_train, categories_train, masks_train, data_test, categories_test, masks_test, charset, charset_cats = load_categories_dataset(args.data)

    num_encoder_tokens = len(charset)
    num_decoder_tokens = len(charset_cats)


if __name__ == '__main__':
    main()