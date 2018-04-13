from __future__ import print_function #pylint bug workaround
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
    parser = argparse.ArgumentParser(description="Wavefront .obj shape sampling and string conversion")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("out_filepath", type=str, help="The output file path in HDF5 format.")
    parser.add_argument("out_grammarpath", type=str, help="The tiling grammar export path in HDF5 format.")
    parser.add_argument('--num_iterations', type=int, metavar='N', default=ITERATIONS, help="Number of iterations for creating random variations out of pairs of objects in the input folder.")
    parser.add_argument("--smiles_column", type=str, default = SMILES_COL_NAME, help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    parser.add_argument('--fix_variations', dest='fix_variations', action='store_true',
                        help='Try to fix local part orientations and remove variations if attempt fails.')

    return parser.parse_args()

def process_folder(folder_name, file_list = []):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            process_folder(subfolfer_name, file_list)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"):
            file_list.append(folder_name + "/" + item_name)       

def augment_folder(file_list=[], word_list=[]):
    for item_id in range(len(file_list) - 1):
        item_name_1 = file_list[item_id]
        sample_id = np.random.randint(item_id, len(file_list))
        item_name_2 = file_list[sample_id]
        current_str = obj_tools.create_variations(item_name_1, item_name_2)
        current_words = current_str.split("\n")                
        for w in current_words:
            word_list.append(str(w))
            #if(len(str(w)) <= MAX_WORD_LENGTH and len(str(w)) > 0):
                #word_list.append(str(w))

def fix_variations(folder_name, exclude_file_list, inputA, inputB):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            fix_variations(subfolfer_name, exclude_file_list, inputA, inputB)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"):
            file_path = folder_name + "/" + item_name
            if file_path != inputA and file_path != inputB and file_path not in exclude_file_list:
                fixed = obj_tools.fix_variation(inputA, inputB, file_path, file_path)
                if fixed != 0:
                    fixed = obj_tools.fix_variation(inputA, inputB, file_path, file_path)
                    if fixed != 0:
                        os.remove(file_path)
                        base_path, extension = os.path.splitext(file_path)
                        os.remove(base_path + ".mtl")

def remove_duplicates(tile_grammar, folder_name, inputA, inputB, word_list = []):

    current_words = []
    for old_str in word_list:
        current_words.append(old_str)

    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            remove_duplicates(tile_grammar, subfolfer_name, inputA, inputB, word_list)
        file_path = folder_name + "/" + item_name
        if  file_path != inputA and file_path != inputB and not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"):
            current_str = obj_tools.obj2string(file_path)
            base_path, extension = os.path.splitext(file_path)
            os.remove(base_path + "_coll_graph.obj")            
            os.remove(base_path + "_coll_graph.mtl")

            if not tile_grammar.check_word(current_str):
                os.remove(file_path)
                os.remove(base_path + ".mtl")
                continue

            current_words.append(current_str)
            for i in range(len(current_words) - 1):
                if tile_grammar.similar_words(current_words[i], current_str):
                    os.remove(file_path)
                    os.remove(base_path + ".mtl")
                    current_words.pop()
                    break

def main():
    args = get_arguments()
    
    initial_file_list = []
    process_folder(args.in_folder, initial_file_list)
    if len(initial_file_list) == 0:
        print("Did not find a valid input file in " + args.in_folder)
        exit()

    if len(initial_file_list) == 1:
        initial_file_list.append(initial_file_list[0])
    else:
        initial_file_list = sorted(initial_file_list)

    inputA = initial_file_list[0]
    inputB = initial_file_list[len(initial_file_list) - 1]

    initial_smiles_strings = []
    initial_smiles_strings.append(str(obj_tools.obj2string(inputA)))
    initial_smiles_strings.append(str(obj_tools.obj2string(inputB)))

    tile_grammar = grammar.TilingGrammar(initial_smiles_strings)
    print("max # neighbors: " + str(tile_grammar.max_degree()))
    tile_grammar.store(args.out_grammarpath)

    print("removing duplicates...")
    remove_duplicates(tile_grammar, args.in_folder, inputA, inputB, initial_smiles_strings)

    smiles_strings = []
    for i in range(args.num_iterations):
        current_file_list = []
        process_folder(args.in_folder, current_file_list)
        print("Current # of variations: " + str(len(current_file_list)))        
        augment_folder(current_file_list, smiles_strings)
        smiles_strings = list(set(smiles_strings))
        if args.fix_variations:
            print("fixing variations...")
            fix_variations(args.in_folder, current_file_list,  inputA, inputB)
        print("removing duplicates...")
        remove_duplicates(tile_grammar, args.in_folder, inputA, inputB, initial_smiles_strings)
        print("Iteration " + str(i) + " # of strings: " + str(len(smiles_strings)))

    loaded_grammar = grammar.TilingGrammar([])
    loaded_grammar.load(args.out_grammarpath)
    
    valid_strings = []
    for w in smiles_strings:
        if(loaded_grammar.check_word(w) == True):
            if len(str(w)) > 0 :
                valid_strings.append(w)      

    print("# valid strings: " + str(len(valid_strings)))
    df = pandas.DataFrame({args.smiles_column : valid_strings})
    df.to_hdf(args.out_filepath, "table", format = "table", data_columns = True)

if __name__ == "__main__":

    main()