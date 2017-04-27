import argparse
import pandas
import h5py
import numpy as np

from sklearn.model_selection import train_test_split

import neuralnets.grammar as grammar

MAX_NUM_ROWS = 500000
SMILES_COL_NAME = "structure"

def get_arguments():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("infile", type=str, help="Input file name")
    parser.add_argument("ingrammar", type=str, help="Grammar file name")
    parser.add_argument("outfile", type=str, help="Output file name")
    parser.add_argument("--length", type=int, metavar="N", default = MAX_NUM_ROWS,
                        help="Maximum number of rows to include (randomly sampled).")
    parser.add_argument("--smiles_column", type=str, default = SMILES_COL_NAME,
                        help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    parser.add_argument("--property_column", type=str,
                        help="Name of the column that contains the property values to predict. Default: None")
    return parser.parse_args()

def main():
    args = get_arguments()
    data = pandas.read_hdf(args.infile, "table")
    keys = data[args.smiles_column].map(len) < 121

    if args.length <= len(keys):
        data = data[keys].sample(n = args.length)
    else:
        data = data[keys]

    loaded_grammar = grammar.TilingGrammar([])
    loaded_grammar.load(args.ingrammar)

    num_data_points = len(data[args.smiles_column])
    vec_dims = len(loaded_grammar.charset) + loaded_grammar.max_degree()

    structures_one_hot = np.zeros((num_data_points, 120, vec_dims))
    for s in range(num_data_points):
        structures_one_hot[s] = loaded_grammar.encode_to_one_hot(data[args.smiles_column][s], 120)

    del data
        
    #data_train, data_test = train_test_split(structures_one_hot, test_size = 0.20)
    train_idx, test_idx = train_test_split(xrange(structures_one_hot.shape[0]), test_size = 0.20)

    h5f = h5py.File(args.outfile, "w")
    h5f.create_dataset("connectivity_dims", data = loaded_grammar.max_degree())
    h5f.create_dataset("charset", data = loaded_grammar.charset)
    h5f.create_dataset("data_train", data = structures_one_hot[train_idx], chunks=(200, 120, vec_dims)) 
    h5f.create_dataset("data_test", data =  structures_one_hot[test_idx], chunks=(200, 120, vec_dims))

    if args.property_column:
        h5f.create_dataset("property_train", data = properties[train_idx])
        h5f.create_dataset("property_test", data = properties[test_idx])
    h5f.close()

if __name__ == "__main__":
    main()
