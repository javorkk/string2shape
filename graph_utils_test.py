import argparse
import os

import obj_tools
import neuralnets.grammar as grammar
import neuralnets.shape_graph as shape_graph
from neuralnets.shape_graph import smiles_variations

def get_arguments():
    parser = argparse.ArgumentParser(description="Shape graph configuration estimation from .obj file collections.")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("-o", "--out_plot", type=str,  help="Where to save the edge configuration plot.")
    return parser.parse_args()

def process_folder(folder_name, file_list = []):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            process_folder(subfolfer_name, file_list)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"):
            file_list.append(folder_name + "/" + item_name)

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
  
    cluster_centers, node_types = shape_graph.categorize_edges(file_list[:100], tile_grammar, args.out_plot)
        
    str_node_ids = str(obj_tools.obj2strings_ids(inputA))
    str_node_ids_list = str_node_ids.split("\n")
    smiles_strings = str_node_ids_list[:len(str_node_ids_list) / 2]
    node_ids_list = str_node_ids_list[len(str_node_ids_list) / 2:]
    
    node_ids = []
    for node_list in node_ids_list:
        node_ids.append([int(i) for i in node_list.split(" ")])

    graph_edges = shape_graph.ShapeGraph(obj_tools.obj2graph(inputA))

    edge_categories = shape_graph.smiles_to_edge_categories(smiles_strings[0], node_ids[0], cluster_centers, graph_edges,  tile_grammar)
    
    print("smiles string len: ", len(smiles_strings[0]))
    print(smiles_strings[0])
    print("edge categories len: ", len(edge_categories))
    print(edge_categories)

    dummy_node_id = len(node_ids[0])
    
    padded_node_ids = []
    num_nodes = 0 
    for char_id, _ in enumerate(smiles_strings[0]):
        if smiles_strings[0][char_id] in tile_grammar.charset:
            padded_node_ids.append(node_ids[0][num_nodes])
            num_nodes += 1
        else:
            padded_node_ids.append(dummy_node_id)
    padded_node_ids.append(dummy_node_id) #ensure at least one occurrence

    smiles_variants, node_lists = smiles_variations(smiles_strings[0], padded_node_ids, tile_grammar, 2)
    print("smiles variants:")
    print(smiles_variants)

    print("node lists:")
    print(node_lists)


    #print("cluster centers:")
    #print(cluster_centers)


if __name__ == "__main__":

    main()
