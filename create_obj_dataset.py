import argparse
import os
import pandas
import obj_tools

SMILES_COL_NAME = 'structure'

def get_arguments():
    parser = argparse.ArgumentParser(description='Wavefront .obj to SMILES string conversion')
    parser.add_argument('in_folder', type=str, help='The folder containing the input .obj files.')
    parser.add_argument('out_filepath', type=str, help='The output file path in HDF5 format.')
    parser.add_argument('--smiles_column', type=str, default = SMILES_COL_NAME, help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    return parser.parse_args()

def main():
    args = get_arguments()
    in_folder = args.in_folder

    smiles_strings = []
    for filename in os.listdir(in_folder):
        if not filename.endswith("_coll_graph.obj") and filename.endswith(".obj"): 
            current_str = obj_tools.obj2string(in_folder + '/' + filename)
            print("Converted " + os.path.join(in_folder, filename) + " to " + current_str)
            smiles_strings.append(current_str)           
            continue
        else:
            continue
    df = pandas.DataFrame({args.smiles_column : smiles_strings})
    df.to_hdf(args.out_filepath, 'df')

if __name__ == '__main__':

    main()