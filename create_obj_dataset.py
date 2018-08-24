from __future__ import print_function #pylint bug workaround
import argparse
import os
import pandas
import re
import obj_tools
import neuralnets.grammar as grammar
import neuralnets.shape_graph as shape_graph
from neuralnets.shape_graph import smiles_variations

SMILES_COL_NAME = "structure"
CATEGORIES_COL_NAME = "edge_categories"
MIN_BOUND_COL_NAME = "min_category"
MAX_BOUND_COL_NAME = "max_category"
MAX_WORD_LENGTH = 1024
VARIATIONS = 1

def get_arguments():
    parser = argparse.ArgumentParser(description="Wavefront .obj to SMILES string conversion")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("out_filepath", type=str, help="The output file path in HDF5 format.")
    parser.add_argument("out_grammarpath", type=str,
                        help="The tiling grammar export path in HDF5 format.")
    parser.add_argument("--smiles_column", type=str, default=SMILES_COL_NAME,
                        help="Column with SMILES strings. Default: %s" % SMILES_COL_NAME)
    parser.add_argument("--categories_column", type=str, default=CATEGORIES_COL_NAME,
                        help="Column with edge categories. Default: %s" % CATEGORIES_COL_NAME)
    parser.add_argument("--plot", type=str, help="Where to save the edge configuration plot.")
    parser.add_argument('--num_variations', type=int, metavar='N', default=VARIATIONS,
                        help="Creates additional variants for each input SMILES string.")
    parser.add_argument('--remove_cycles', dest='remove_cycles', action='store_true',
                        help='Convert input graphs to trees by discarding edges.')
    return parser.parse_args()

#def process_folder(folder_name, word_list = []):
#    for item_name in os.listdir(folder_name):
#        subfolfer_name = os.path.join(folder_name, item_name)
#        if os.path.isdir(subfolfer_name):
#            process_folder(subfolfer_name, word_list)
#        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"):
#            current_str = obj_tools.obj2strings(folder_name + "/" + item_name)
#            current_words = current_str.split("\n")
#            print("Converted " + os.path.join(folder_name, item_name) + " to " + current_words[0])
#            for w in current_words:
#                if(len(str(w)) <= MAX_WORD_LENGTH):
#                    word_list.append(str(w))

def process_folder(folder_name, file_list=[]):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            file_list.append(process_folder(subfolfer_name, file_list))
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"):
            file_list.append(folder_name + "/" + item_name)
    return file_list

def main():
    args = get_arguments()
    file_list = process_folder(args.in_folder)
    file_list = sorted(file_list)

    input_a = file_list[0]
    input_b = file_list[len(file_list) - 1]

    initial_smiles_strings = []
    initial_smiles_strings.append(str(obj_tools.obj2string(input_a)))
    initial_smiles_strings.append(str(obj_tools.obj2string(input_b)))

    tile_grammar = grammar.TilingGrammar(initial_smiles_strings)  

    cluster_centers, _ = shape_graph.categorize_edges(file_list[:100], tile_grammar, args.plot)

    num_categories = 0
    categories_prefix = [0]
    for clusters in cluster_centers:
        num_categories += clusters.shape[0]
        categories_prefix.append(num_categories)
    
    tile_grammar.set_categories_prefix(categories_prefix)
    tile_grammar.store(args.out_grammarpath)

    smiles_strings = []
    edge_categories = []
    edge_cat_min = []
    edge_cat_max = []

    for file_name in file_list:
        str_node_ids = str(obj_tools.obj2strings_ids(file_name))
        if str_node_ids == '':
            continue
        str_node_ids_list = str_node_ids.split("\n")
        initial_strings = str_node_ids_list[:len(str_node_ids_list) / 2]
        node_ids_list = str_node_ids_list[len(str_node_ids_list) / 2:]

        current_strings = []
        if args.remove_cycles:
            for elem in initial_strings:
                current_strings.append(re.sub("["+tile_grammar.DIGITS + tile_grammar.NUM_DELIMITER+"]", "", elem))
        else:
            current_strings = initial_strings

        node_ids = []
        for node_list in node_ids_list:
            node_ids.append([int(i) for i in node_list.split(" ")])

        graph_edges = shape_graph.ShapeGraph(obj_tools.obj2graph(file_name))

        for i, _ in enumerate(current_strings):
            dummy_node_id = len(node_ids[0])

            padded_node_ids = []
            num_nodes = 0 
            for char_id, _ in enumerate(current_strings[i]):
                if current_strings[i][char_id] in tile_grammar.charset:
                    padded_node_ids.append(node_ids[0][num_nodes])
                    num_nodes += 1
                else:
                    padded_node_ids.append(dummy_node_id)
            padded_node_ids.append(dummy_node_id) #ensure at least one occurrence

            variant_strings, variant_nodes = smiles_variations(current_strings[i], padded_node_ids, tile_grammar, args.num_variations)
            for word, padded_nodes in zip(variant_strings, variant_nodes):
                nodes = [x for x in padded_nodes if x != dummy_node_id]
                if not args.remove_cycles and not tile_grammar.check_word(word):
                    continue
                if len(str(word)) <= MAX_WORD_LENGTH and len(str(word)) > 0 and word not in smiles_strings:
                    smiles_strings.append(word)
                    current_categories = shape_graph.smiles_to_edge_categories(word, nodes, cluster_centers, graph_edges, tile_grammar)
                    categories_str = ""
                    for cat in current_categories:
                        categories_str += str(cat) + " "
                    edge_categories.append(categories_str[:-1])

                    if len(current_categories) > len(word):
                        print("wrong number of edge categories: ", len(current_categories), " instead of ", len(word))
                        print(word)
                        print(current_categories)

                    category_bounds = tile_grammar.smiles_to_categories_bounds(word)
                    min_bound_str = ""
                    max_bound_str = ""
                    for bounds in category_bounds:
                        min_bound_str += str(bounds[0]) + " "
                        max_bound_str += str(bounds[1]) + " "
                    edge_cat_min.append(min_bound_str[:-1])
                    edge_cat_max.append(max_bound_str[:-1])


    print("# items: " + str(len(smiles_strings)))

    df = pandas.DataFrame({args.smiles_column       : smiles_strings,
                           args.categories_column   : edge_categories,
                           MIN_BOUND_COL_NAME       : edge_cat_min, 
                           MAX_BOUND_COL_NAME       : edge_cat_max
                           })
    df.to_hdf(args.out_filepath, "table", format="table", data_columns=True)

if __name__ == "__main__":

    main()
    