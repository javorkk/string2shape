import argparse
import os
import re
import obj_tools

from neuralnets import grammar

def get_arguments():
    parser = argparse.ArgumentParser(description="SMILES string to Wavefront .obj conversion by file search.")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("in_grammar", type=str, help="The tiling grammar for the set of models.")
    parser.add_argument("in_string", type=str, help="The smiles string encoding the graph to search for.")
    return parser.parse_args()

def str_to_file(folder_name, query_word, tiling_grammar):
    best_match_w = ""
    best_match_f = ""
    best_similarity = 1.0
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            str_to_file(subfolfer_name, query_word, tiling_grammar)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"): 
            #current_str = obj_tools.obj2string(folder_name + "/" + item_name)
            current_strings = obj_tools.obj2strings(folder_name + "/" + item_name).split("\n")

            for current_str_1 in current_strings:
                current_str = str(current_str_1)
                if tiling_grammar.word_similarity(query_word, current_str) < best_similarity:
                    best_match_f = item_name
                    best_match_w = current_str
                    best_similarity = tiling_grammar.word_similarity(query_word, current_str)

                mismatch = False
                # for i in range(len(tiling_grammar.DIGITS)):
                #     if(query_word.count(tiling_grammar.DIGITS[i]) != current_str.count(tiling_grammar.DIGITS[i])):
                #         mismatch = True
                #         break#different number of cycles
                for i in range(1, len(tiling_grammar.charset)):
                    if(query_word.count(tiling_grammar.charset[i]) != current_str.count(tiling_grammar.charset[i])):
                        mismatch = True
                        break

                if not mismatch:
                    print(query_word + " found in " + item_name)
                    return True
    print("Best match: ", best_match_w)
    print("Found in: ", best_match_f)
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