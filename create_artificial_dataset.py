import argparse
import os
import pandas
import obj_tools
import neuralnets.grammar as grammar
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

    #check for cycles
    if(word.find("0") != -1 and word.find("A") != -1):
        word_list.append(word)
        if(numpy.random.random(1) < 0.5):
            variant = word.replace("A", "AA")
            str_len_variation_generator(word_list, variant, max_length, char_pos, depth + 1)
        return

    if(len(word) <= char_pos or depth >= 1 and numpy.random.random(1) < 0.75):
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
    if(len(word) <= char_pos):
        word_list.append(word)
        return

    if(depth >= word.count("C") + word.count("D")):
        word_list.append(word)
        return

    if(depth >= 2 and numpy.random.random(1) < 0.5):
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

def process_folder(folder_name, word_list = []):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            process_folder(subfolfer_name, word_list)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"): 
            current_str = obj_tools.obj2strings(folder_name + "/" + item_name)
            current_words = current_str.split("\n")
            print("Converted " + os.path.join(folder_name, item_name) + " to " + current_words[0])
            for w in current_words:
                if(len(str(w)) <= MAX_WORD_LENGTH):
                    word_list.append(str(w)) 
def main():
    args = get_arguments()

    initial_smiles_strings = []
    process_folder(args.in_folder, initial_smiles_strings)
    initial_smiles_strings = list(set(initial_smiles_strings))
    print("# initial strings: " + str(len(initial_smiles_strings)))
    
    length_variations = []
    for word in initial_smiles_strings:
        str_len_variation_generator(length_variations, word)

    print("# length variations: " + str(len(length_variations)))

    char_variations = []
    for word in length_variations:
        str_char_variation_generator(char_variations, word)

    print("# char variations: " + str(len(char_variations)))
    
    tile_grammar = grammar.TilingGrammar(initial_smiles_strings)
    output_strings = []
    for word in char_variations:
        if(tile_grammar.check_word(word) == True):
            output_strings.append(word)
    
    tile_grammar.store(args.out_grammarpath)
    print("# all valid variations: " + str(len(char_variations)))

    df = pandas.DataFrame({args.smiles_column : output_strings})
    df.to_hdf(args.out_filepath, "table", format = "table", data_columns = True)

if __name__ == "__main__":

    main()