import argparse
import os
import pandas
import obj_tools
import grammar

SMILES_COL_NAME = "structure"
MAX_WORD_LENGTH = 120

def get_arguments():
    parser = argparse.ArgumentParser(description="Wavefront .obj to SMILES string conversion")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("out_filepath", type=str, help="The output file path in HDF5 format.")
    parser.add_argument("out_grammarpath", type=str, help="The tiling grammar export path in HDF5 format.")
    parser.add_argument("--smiles_column", type=str, default = SMILES_COL_NAME, help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    return parser.parse_args()

def main():
    args = get_arguments()
    in_folder = args.in_folder

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

    tile_grammar = grammar.TilingGrammar(initial_smiles_strings)
    tile_grammar.store(args.out_grammarpath)
    loaded_grammar = grammar.TilingGrammar([])
    loaded_grammar.load(args.out_grammarpath)
    for w in initial_smiles_strings:
        if(loaded_grammar.check_word(w) == False):
            print("Wrongly detected as invalid " + w)

    print("# items: " + str(len(initial_smiles_strings)))
    df = pandas.DataFrame({args.smiles_column : initial_smiles_strings})
    #df.to_hdf(args.out_filepath, "table", format = "table", data_columns = True)

if __name__ == "__main__":

    main()