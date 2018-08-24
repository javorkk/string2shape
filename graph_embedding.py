from __future__ import print_function
import argparse
import os
import numpy as np
import random
import sys

import obj_tools
import neuralnets.shape_graph as shape_graph

from neuralnets.seq2seq import Seq2SeqAE, Seq2SeqRNN, Seq2SeqNoMaskRNN
from neuralnets.grammar import TilingGrammar
from neuralnets.utils import load_categories_dataset, decode_smiles_from_indexes, from_one_hot_array
from neuralnets.shape_graph import ShapeGraph, smiles_variations, categorize_edges
from train_seq2seq import decode_sequence_rnn


from collections import Counter

def most_common_elem(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


LSTM_SIZE = 512
NUM_ATTEMPTS = 128

TREE_GRAMMAR = True

def get_arguments():
    parser = argparse.ArgumentParser(description='Shape embedding in 3D from strings')
    parser.add_argument('in_folder', type=str, help='Folder with example objects.')
    parser.add_argument('model', type=str, help='The trained seq2seq model.')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('grammar', type=str, help='The HDF5 file with the tiling grammar.')
    parser.add_argument('out', type=str, help='Where to save the output file. If this file exists, it will be overwritten.')
    parser.add_argument('in_word', type=str, help='Target SMILES string to embed.')
    parser.add_argument('--num_attempts', type=int, default=NUM_ATTEMPTS, help='Number of attempts.')
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
                    num_variants=128):

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
    complete_edge_list = []
    for edge in edge_lists[0]:
        if edge[0] != dummy_node_id and edge[1] != dummy_node_id and edge[0] != edge[1]:
            complete_edge_list.append(edge)
            complete_edge_list.append([edge[1], edge[0]])

    #category_bounds_iedges = grammar.smiles_to_categories_bounds(input_word, invert_edge_direction=True)

    num_categories = grammar.categories_prefix[-1] + 1

    node_category_pairs = set()
    output_sequence = []
    per_edge_categories = []
    for edge_id, edge in enumerate(complete_edge_list):
        local_categories = []
        node_id = edge[0]
        for j in range(0, num_variants):
            if edge in edge_lists[j]: #edge direction can be reversed in the other list
                idx = edge_lists[j].index(edge)
                category = decoded_sequences[j][idx]
                if (node_id, category) not in node_category_pairs:
                    local_categories.append(category)
        if len(local_categories) == 0:
            local_categories.append(num_categories)
            # i_edge_id = edge_id - len(complete_edge_list) / 2
            # category_bounds = category_bounds_iedges[i_edge_id]
            # allowed_categories = [x for x in range(category_bounds[0], category_bounds[1]) if (node_id, x) not in node_category_pairs]
            # local_categories.append(random.choice(allowed_categories))
        per_edge_categories.append(local_categories)
        best_category = most_common_elem(local_categories)
        output_sequence.append(best_category)
        if best_category != num_categories and (node_id, best_category) in node_category_pairs:
            print("repeated node, category pair: ", node_id, best_category)
        node_category_pairs.add((node_id, best_category))

    return output_sequence, complete_edge_list

def file_to_graph_with_categories(filename, cluster_centers, tile_grammar):
    str_node_ids_a = str(obj_tools.obj2strings_ids(filename))
    str_node_ids_list_a = str_node_ids_a.split("\n")
    smiles_strings_a = str_node_ids_list_a[:len(str_node_ids_list_a) / 2]
    node_ids_list_a = str_node_ids_list_a[len(str_node_ids_list_a) / 2:]
    
    node_ids_a = []
    for node_list in node_ids_list_a:
        node_ids_a.append([int(i) for i in node_list.split(" ") if i != ""])

    graph_edges_a = shape_graph.ShapeGraph(obj_tools.obj2graph(filename))

    all_edge_categories_a, all_edges_a = shape_graph.smiles_to_all_edge_categories(smiles_strings_a[0], node_ids_a[0], cluster_centers, graph_edges_a,  tile_grammar)

    if len(all_edge_categories_a) != len(all_edges_a):
        print("Error, mismatching number of edges",len(all_edges_a),"and edge categories", len(all_edge_categories_a))

    return all_edge_categories_a, all_edges_a


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
    if os.path.isfile(args.grammar):
        tile_grammar.load(args.grammar)
    else:
        raise ValueError("Grammar file %s doesn't exist" % args.grammar)

    if TREE_GRAMMAR:
        tile_grammar.convert_to_tree_grammar()

    cluster_centers, node_types = shape_graph.categorize_edges(file_list[:100], tile_grammar)

    all_edge_categories_a, all_edges_a =  file_to_graph_with_categories(inputA, cluster_centers, tile_grammar)

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

    category_pairs = set()
    for edge, cat in zip(all_edges_a, all_edge_categories_a):
        reverse_edge = [edge[1], edge[0]]
        reverse_cat  = all_edge_categories_a[all_edges_a.index(reverse_edge)]
        category_pairs.add((cat, reverse_cat))

    all_edge_categories_b, all_edges_b = file_to_graph_with_categories(inputB, cluster_centers, tile_grammar)

    output_str_b = ""
    for edge in all_edges_b:
        output_str_b += str(edge[0]) + " "
    output_str_b += "\n"
    for edge in all_edges_b:
        output_str_b += str(edge[1]) + " "
    output_str_b += "\n"
    for categ in all_edge_categories_b:
        output_str_b += str(categ) + " "
    output_str_b += "\n"
    
    for edge, cat in zip(all_edges_b, all_edge_categories_b):
        reverse_edge = [edge[1], edge[0]]
        reverse_cat  = all_edge_categories_b[all_edges_b.index(reverse_edge)]
        category_pairs.add((cat, reverse_cat))

    data_train, categories_train, masks_train, data_test, categories_test, masks_test, charset, charset_cats = load_categories_dataset(args.data)

    num_encoder_tokens = len(charset)
    num_decoder_tokens = len(charset_cats)

    model = Seq2SeqRNN()
    if os.path.isfile(args.model):
        model.load(charset, charset_cats, args.model, lstm_size=LSTM_SIZE)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    # setup toolbar
    sys.stdout.write("[%s]" % (" " * args.num_attempts))
    sys.stdout.flush()
    sys.stdout.write("\b" * (args.num_attempts+1)) # return to start of line, after '['



    for num_attempts in range(0,args.num_attempts):
        target_edge_categories, target_edges = decode_graph(model, tile_grammar, charset, args.in_word, max_length=data_train.shape[1], num_variants=32)

        # for edge, cat in zip(target_edges, target_edge_categories):
        #     reverse_edge = [edge[1], edge[0]]
        #     reverse_cat  = target_edge_categories[target_edges.index(reverse_edge)]
        #     if (cat, reverse_cat) not in category_pairs:
        #         for pair in category_pairs:
        #             if pair[0] == cat:
        #                 node_id = edge[1]
        #                 per_node_cats = [edge_cat[1] for edge_cat in zip(target_edges, target_edge_categories) if edge_cat[0][0] == node_id]
        #                 if pair[1] not in per_node_cats:
        #                     target_edge_categories[target_edges.index(reverse_edge)] = pair[1]
        #                     break
        #             elif pair[1] == reverse_cat:
        #                 if pair[0] == cat:
        #                     node_id = edge[0]
        #                     per_node_cats = [edge_cat[1] for edge_cat in zip(target_edges, target_edge_categories) if edge_cat[0][0] == node_id]
        #                     if pair[0] not in per_node_cats:
        #                         target_edge_categories[target_edges.index(edge)] = pair[0]
        #                         break

        #target_edge_categories, target_edges = file_to_graph_with_categories(random.choice(file_list), cluster_centers, tile_grammar)

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

        #target_str = output_str_a + output_str_b + output_str_b

        filename, ext = os.path.splitext(args.out)
        filename += "_" + str(num_attempts)
        result = obj_tools.string2obj(inputA, inputB, target_str, filename)
        if result == 0:
            sys.stdout.write("\n")
            print("Successfull attempt with target string: ")
            print(target_str)
            break
        elif result == 1:
            sys.stdout.write("\n")
            print("Successfull embedding not strictly according to the target string: ")
            print(target_str)
            break

        sys.stdout.write("#")
        sys.stdout.flush()

sys.stdout.write("\n")

if __name__ == '__main__':
    main()