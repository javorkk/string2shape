import argparse
import os
import pandas
import obj_tools
import neuralnets.grammar as grammar
import neuralnets.shape_graph as shape_graph

SMILES_COL_NAME = "structure"
CATEGORIES_COL_NAME = "edge_categories"
MAX_WORD_LENGTH = 120

def get_arguments():
    parser = argparse.ArgumentParser(description="Wavefront .obj to SMILES string conversion")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("out_filepath", type=str, help="The output file path in HDF5 format.")
    parser.add_argument("out_grammarpath", type=str, help="The tiling grammar export path in HDF5 format.")
    parser.add_argument("--smiles_column", type=str, default = SMILES_COL_NAME, help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    parser.add_argument("--categories_column", type=str, default = CATEGORIES_COL_NAME, help="Name of the column that contains edge categories. Default: %s" % CATEGORIES_COL_NAME)
    parser.add_argument("--edge_types_plot", type=str,  help="Where to save the edge configuration plot.")
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
    tile_grammar.store(args.out_grammarpath)

    cluster_centers, node_types = shape_graph.categorize_edges(file_list[:100], tile_grammar, args.edge_types_plot)

    smiles_strings = []
    edge_categories = []

    for file_name in file_list:
        str_node_ids = str(obj_tools.obj2strings_ids(file_name))
        if str_node_ids == '':
            continue
        str_node_ids_list = str_node_ids.split("\n")
        current_strings = str_node_ids_list[:len(str_node_ids_list) / 2]
        node_ids_list = str_node_ids_list[len(str_node_ids_list) / 2:]

        node_ids = []
        for node_list in node_ids_list:
            node_ids.append([int(i) for i in node_list.split(" ")])
    
        graph_edges = shape_graph.ShapeGraph(obj_tools.obj2graph(file_name))
    
        for i in range(len(current_strings)):
            w = current_strings[i]
            nodes = node_ids[i]
            if tile_grammar.check_word(w) == True and len(str(w)) <= MAX_WORD_LENGTH and len(str(w)) > 0 :
                smiles_strings.append(w)
                current_categories = shape_graph.smiles_to_edge_categories(w, nodes, cluster_centers, graph_edges, tile_grammar)
                categories_str = ""
                for cat in current_categories:
                    categories_str += str(cat) + " "
                edge_categories.append(categories_str[:-1])

    print("# items: " + str(len(smiles_strings)))

    df = pandas.DataFrame({args.smiles_column : smiles_strings, args.categories_column : edge_categories})
    df.to_hdf(args.out_filepath, "table", format = "table", data_columns = True)

if __name__ == "__main__":

    main()