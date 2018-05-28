from __future__ import print_function
import argparse
import os
import numpy as np

import obj_tools
import neuralnets.shape_graph as shape_graph

from neuralnets.seq2seq import Seq2SeqAE, Seq2SeqRNN, Seq2SeqNoMaskRNN
from neuralnets.grammar import TilingGrammar
from neuralnets.utils import load_categories_dataset, decode_smiles_from_indexes, from_one_hot_array
from neuralnets.shape_graph import ShapeGraph, smiles_variations, categorize_edges
from train_seq2seq import decode_sequence_rnn

LSTM_SIZE = 292

def get_arguments():
    parser = argparse.ArgumentParser(description='Sequence to sequence autoencoder network')
    parser.add_argument('in_folder', type=str, help='Folder with example objects.')
    parser.add_argument('model', type=str, help='The trained seq2seq model.')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('grammar', type=str, help='The HDF5 file with the tiling grammar.')
    parser.add_argument('out', type=str, help='Where to save the output file. If this file exists, it will be overwritten.')
    parser.add_argument('in_word', type=str, help='Target SMILES string to embed.')
    return parser.parse_args()

def process_folder(folder_name, file_list = []):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            process_folder(subfolfer_name, file_list)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"):
            file_list.append(folder_name + "/" + item_name)

def decode_graph(model,
                    grammar,
                    input_charset,
                    input_word,
                    max_length=120,
                    num_variants=10):

    if num_variants <= 1:
        num_variants = 1 
    ##############################################################################################################
    #Generate multiple string variants for the input graph
    ##############################################################################################################
    padded_node_ids = []
    num_nodes = 0 
    for char_id, _ in enumerate(input_word):
        if input_word[char_id] in grammar.charset:
            padded_node_ids.append(num_nodes)
            num_nodes += 1
        else:
            padded_node_ids.append(max_length)

    dummy_node_id = num_nodes

    for i, _ in enumerate(padded_node_ids):
        if padded_node_ids[i] == max_length:
            padded_node_ids[i] = dummy_node_id

    padded_node_ids.append(dummy_node_id) #ensure at least one occurrence

    smiles_variants, node_variants = smiles_variations(input_word, padded_node_ids, grammar, num_variants - 1)

    smiles_strings = [input_word] + smiles_variants
    node_lists = [padded_node_ids] + node_variants
    edge_lists = []
    for word, nodes in zip(smiles_strings, node_lists):
        edge_lists.append(grammar.smiles_to_edges(word, nodes))


    input_sequences = np.empty(dtype='float32', shape=(num_variants, max_length, len(input_charset)))
    input_masks = np.empty(dtype='float32', shape=(num_variants, max_length, grammar.categories_prefix[-1] + 1))
    for i, word in enumerate(smiles_strings):
        input_sequences[i] = grammar.smiles_to_one_hot(word.ljust(max_length), input_charset)
        input_masks[i] = grammar.smiles_to_mask(word, max_length)

    ##############################################################################################################
    #Classify each string (estimate edge configurations)
    ##############################################################################################################
    output_charset = list(range(0, grammar.categories_prefix[-1] + 1, 1))

    decoded_sequences = []
    for i in range(num_variants):
        decoded_sequences.append(decode_sequence_rnn(model, input_sequences[i:i+1], len(smiles_strings[i]), output_charset, input_masks[i:i+1]))

    complete_edge_list = edge_lists[0]
    for edge in edge_lists[0]:
        complete_edge_list.append([edge[1], edge[0]])

    output_sequence = []
    per_edge_categories = []
    for edge_id, edge in enumerate(complete_edge_list):
        local_categories = [decoded_sequences[0][edge_id]]
        if edge[0] != dummy_node_id or edge[1] != dummy_node_id:
            for j in range(1, num_variants):
                if edge in edge_lists[j]: #edge direction can be reversed in the other list
                    idx = edge_lists[j].index(edge)
                    local_categories.append(decoded_sequences[j][idx])
        per_edge_categories.append(local_categories)
        output_sequence.append(most_common_elem(local_categories))

    return output_sequence, complete_edge_list

def file_to_graph_with_categories(filename, cluster_centers, tile_grammar):
    str_node_ids_a = str(obj_tools.obj2strings_ids(filename))
    str_node_ids_list_a = str_node_ids_a.split("\n")
    smiles_strings_a = str_node_ids_list_a[:len(str_node_ids_list_a) / 2]
    node_ids_list_a = str_node_ids_list_a[len(str_node_ids_list_a) / 2:]
    
    node_ids_a = []
    for node_list in node_ids_list_a:
        node_ids_a.append([int(i) for i in node_list.split(" ")])

    graph_edges_a = shape_graph.ShapeGraph(obj_tools.obj2graph(filename))
    edge_categories_a = shape_graph.smiles_to_edge_categories(smiles_strings_a[0], node_ids_a[0], cluster_centers, graph_edges_a,  tile_grammar)

    all_edge_categories_a, all_edges_a = shape_graph.smiles_to_all_edge_categories(smiles_strings_a[0], node_ids_a[0], cluster_centers, graph_edges_a,  tile_grammar)

    if len(all_edge_categories_a) != len(all_edges_a):
        print("Error, mismatching number of edges",len(all_edges_a),"and edge categories", len(all_edge_categories_a))

    output_str_a = ""
    for edge in all_edges_a:
        output_str_a += str(edge[0]) + " "
    output_str_a += "\n"
    for edge in all_edges_a:
        output_str_a += str(edge[1]) + " "
    output_str_a += "\n"
    for categ in all_edge_categories_a:
        output_str_a += str(categ) + " "
    output_str_a += "\n"

    return output_str_a


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

    cluster_centers, node_types = shape_graph.categorize_edges(file_list[:10], tile_grammar)

    output_str_a =  file_to_graph_with_categories(inputA, cluster_centers, tile_grammar)
    print("input string a: ")
    print(output_str_a)

    output_str_b =  file_to_graph_with_categories(inputB, cluster_centers, tile_grammar)
    print("input string b: ")
    print(output_str_b)

    data_train, categories_train, masks_train, data_test, categories_test, masks_test, charset, charset_cats = load_categories_dataset(args.data)

    num_encoder_tokens = len(charset)
    num_decoder_tokens = len(charset_cats)

    model = Seq2SeqRNN()
    if os.path.isfile(args.model):
        model.load(charset, charset_cats, args.model, lstm_size=LSTM_SIZE)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    target_edge_categories, target_edges = decode_graph(model, tile_grammar, charset, args.in_word, max_length=data_train.shape[1], num_variants=32)

    target_str = output_str_a + output_str_b
    for edge in target_edges:
        target_str += str(edge[0]) + " "
    target_str += "\n"
    for edge in target_edges:
        target_str += str(edge[1]) + " "
    target_str += "\n"
    for categ in target_edge_categories:
        target_str += str(categ) + " "
    target_str += "\n"

    filename, ext = os.path.splitext(args.out)


    obj_tools.string2obj(inputA, inputB, target_str, filename)

if __name__ == '__main__':
    main()