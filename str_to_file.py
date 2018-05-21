import argparse
import os
import obj_tools

from neuralnets import grammar

def get_arguments():
    parser = argparse.ArgumentParser(description="SMILES string to Wavefront .obj conversion by file search.")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("in_grammar", type=str, help="The tiling grammar for the set of models.")
    parser.add_argument("in_string", type=str, help="The smiles string encoding the graph to search for.")
    return parser.parse_args()

def str_to_file(folder_name, query_word, tiling_grammar):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            str_to_file(subfolfer_name, query_word, tiling_grammar)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"): 
            current_str = obj_tools.obj2string(folder_name + "/" + item_name)
            if tiling_grammar.similar_words(query_word, current_str):
                print(query_word + " found in " + item_name)
                return True
    return False

def main():
    args = get_arguments()

    in_smiles_string = args.in_string

    tiling_grammar = grammar.TilingGrammar([])
    if os.path.isfile(args.in_grammar):
        tiling_grammar.load(args.in_grammar)
    else:
        raise ValueError("Grammar file %s doesn't exist" % args.in_grammar)

    success = str_to_file(args.in_folder, in_smiles_string, tiling_grammar)
    if not success:
        print("Did not find " + in_smiles_string)

if __name__ == "__main__":

    main()