import argparse
import os
import pandas
import obj_tools
import grammar
import numpy

SMILES_COL_NAME = "structure"
MAX_WORD_LENGTH = 120

def get_arguments():
    parser = argparse.ArgumentParser(description="Wavefront .obj to SMILES string conversion")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("out_filepath", type=str, help="The output file path in HDF5 format.")
    parser.add_argument("out_grammarpath", type=str, help="The tiling grammar export path in HDF5 format.")
    parser.add_argument("--smiles_column", type=str, default = SMILES_COL_NAME, help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    return parser.parse_args()

def str_len_variation_generator(word_list, word, max_length = MAX_WORD_LENGTH, char_pos = 0, depth = 0):
    if(len(word) > max_length):
        return
    if(len(word) <= char_pos or depth >= 2 and numpy.random.random(1) < 0.5):
        word_list.append(word)
        return

    if(word[char_pos] == "A"):
        str_len_variation_generator(word_list, word, max_length, char_pos + 1, depth)
        variant = word[0:char_pos] + "A" + word[char_pos:]
        str_len_variation_generator(word_list, variant, max_length, char_pos + 2, depth + 1)
    else:
        str_len_variation_generator(word_list, word, max_length, char_pos + 1, depth)
    return

def str_char_variation_generator(word_list, word, max_length = MAX_WORD_LENGTH, char_pos = 0, depth = 0):
    if(len(word) <= char_pos or depth >= 3):
        word_list.append(word)
        return

    if(word[char_pos] == "C"):
        str_char_variation_generator(word_list, word, max_length, char_pos + 1, depth + 1)
        variant = word[0:char_pos] + "D" + word[char_pos + 1:] 
        str_char_variation_generator(word_list, variant, max_length, char_pos + 1, depth + 1)
    elif(word[char_pos] == "D"):
        str_char_variation_generator(word_list, word, max_length, char_pos + 1, depth + 1)
        variant = word[0:char_pos] + "C" + word[char_pos + 1:]
        str_char_variation_generator(word_list, variant, max_length, char_pos + 1, depth + 1)
    else:
        str_char_variation_generator(word_list, word, max_length, char_pos + 1, depth)
    return

def main():
    args = get_arguments()
    in_folder = args.in_folder

    #test_strings = []
    #str_len_variation_generator(test_strings, "A(AAAAB(AD)D)B(AB(AB(C)AB(AAAAB(AD)AC)AAAAAB(AD)AC)C)AAAAAB(D)AD")
    #final_string = []
    #for word in test_strings:
    #    str_char_variation_generator(final_string, word)    
    #for word in final_string:
    #    print(word)

    initial_smiles_strings = []
    for filename in os.listdir(in_folder):
        if not filename.endswith("_coll_graph.obj") and filename.endswith(".obj"): 
            current_str = obj_tools.obj2string(in_folder + "/" + filename)
            print("Converted " + os.path.join(in_folder, filename) + " to " + current_str)
            if(len(str(current_str)) <= MAX_WORD_LENGTH):
                initial_smiles_strings.append(str(current_str))           
            continue
        else:
            continue
    print("# initial strings: " + str(len(initial_smiles_strings)))
    
    lengh_variations = []
    for word in initial_smiles_strings:
        str_len_variation_generator(lengh_variations, word)

    print("# length variations: " + str(len(lengh_variations)))

    char_variations = []
    for word in lengh_variations:
        str_char_variation_generator(char_variations, word)

    print("# char variations: " + str(len(char_variations)))
    
    tile_grammar = grammar.TilingGrammar(initial_smiles_strings)
    output_strings = []
    for word in char_variations:
        if(tile_grammar.check_word(word) == True):
            output_strings.append(word)
    
    print("# all valid variations: " + str(len(char_variations)))

    df = pandas.DataFrame({args.smiles_column : output_strings})
    df.to_hdf(args.out_filepath, "table", format = "table", data_columns = True)

if __name__ == "__main__":

    main()