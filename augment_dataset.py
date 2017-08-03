import argparse
import os
import numpy as np
import pandas
import obj_tools
import neuralnets.grammar as grammar

SMILES_COL_NAME = "structure"
MAX_WORD_LENGTH = 120
ITERATIONS = 2

def get_arguments():
    parser = argparse.ArgumentParser(description="Wavefront .obj to SMILES string conversion")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("out_filepath", type=str, help="The output file path in HDF5 format.")
    parser.add_argument("out_grammarpath", type=str, help="The tiling grammar export path in HDF5 format.")
    parser.add_argument('--num_iterations', type=int, metavar='N', default=ITERATIONS, help='Number of iterations for creating random variations out of pairs of objects in the input folder.')
    parser.add_argument("--smiles_column", type=str, default = SMILES_COL_NAME, help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    return parser.parse_args()

def process_folder(folder_name, file_list = []):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            process_folder(subfolfer_name, word_list)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"):
            file_list.append(folder_name + "/" + item_name)       

def augment_folder(file_list=[], word_list=[]):
    for item_id in range(len(file_list) - 1):
        item_name_1 = file_list[item_id]
        sample_id = np.random.randint(item_id + 1, len(file_list))
        item_name_2 = file_list[sample_id]
        current_str = obj_tools.create_variations(item_name_1, item_name_2)
        current_words = current_str.split("\n")                
        for w in current_words:
            if(len(str(w)) <= MAX_WORD_LENGTH and len(str(w)) > 0):
                word_list.append(str(w))

def main():
    args = get_arguments()

    initial_smiles_strings = []
    for i in range(args.num_iterations):
        initial_file_list = []
        process_folder(args.in_folder, initial_file_list)
        initial_smiles_strings = []
        augment_folder(initial_file_list, initial_smiles_strings)
        initial_smiles_strings = list(set(initial_smiles_strings))
        print("Iteration " + str(i) +" # of strings: " + str(len(initial_smiles_strings)))

    tile_grammar = grammar.TilingGrammar(initial_smiles_strings)
    print("max # neighbors: " + str(tile_grammar.max_degree()))
    tile_grammar.store(args.out_grammarpath)
    #loaded_grammar = grammar.TilingGrammar([])
    #loaded_grammar.load(args.out_grammarpath)
    #for w in initial_smiles_strings:
    #    if(loaded_grammar.check_word(w) == False):
    #        print("Wrongly detected as invalid " + w)
    #    one_hot_vec = loaded_grammar.encode_to_one_hot(w, 120)
    #    valid_vec = loaded_grammar.check_one_hot(one_hot_vec)
    #    if not valid_vec:
    #        print("Wrongly detected as invalid ")
    #        print(loaded_grammar.print_one_hot(one_hot_vec))        

    print("# items: " + str(len(initial_smiles_strings)))
    df = pandas.DataFrame({args.smiles_column : initial_smiles_strings})
    df.to_hdf(args.out_filepath, "table", format = "table", data_columns = True)

if __name__ == "__main__":

    main()