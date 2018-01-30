import argparse
import pandas
import h5py
import numpy as np
from neuralnets.utils import one_hot_array, one_hot_index

from sklearn.model_selection import train_test_split

MAX_NUM_ROWS = 500000
SMILES_COL_NAME = 'structure'
CATEGORIES_COL_NAME = "edge_categories"
MAX_WORD_LENGTH = 120

def get_arguments():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('infile', type=str, help='Input file name')
    parser.add_argument('outfile', type=str, help='Output file name')
    parser.add_argument('--length', type=int, metavar='N', default = MAX_NUM_ROWS,
                        help='Maximum number of rows to include (randomly sampled).')
    parser.add_argument('--smiles_column', type=str, default = SMILES_COL_NAME,
                        help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    parser.add_argument("--categories_column", type=str, default = CATEGORIES_COL_NAME,
                        help="Name of the column that contains edge categories. Default: %s" % CATEGORIES_COL_NAME)
    parser.add_argument('--property_column', type=str,
                        help="Name of the column that contains the property values to predict. Default: None")
    return parser.parse_args()

def chunk_iterator(dataset, chunk_size=200):
    chunk_indices = np.array_split(np.arange(len(dataset)),
                                    len(dataset)/chunk_size)
    for chunk_ixs in chunk_indices:
        chunk = dataset[chunk_ixs]
        yield (chunk_ixs, chunk)
    raise StopIteration

def main():
    args = get_arguments()
    data = pandas.read_hdf(args.infile, 'table')
    keys = data[args.smiles_column].map(len) < MAX_WORD_LENGTH + 1

    if args.length <= len(keys):
        data = data[keys].sample(n=args.length)
    else:
        data = data[keys]

    structures = data[args.smiles_column].map(lambda x: list(x.ljust(MAX_WORD_LENGTH)))

    edge_categories = np.empty(dtype=int, shape=(0, MAX_WORD_LENGTH))
    if args.categories_column in data.keys():
        edge_categories_lists = data[args.categories_column].map(lambda x: [int(c) for c in x.split(" ") ])
        max_category = max(max(edge_categories_lists))

        edge_categories = np.append(edge_categories, np.zeros((len(edge_categories_lists), MAX_WORD_LENGTH), dtype=int), axis = 0)
        for i, _ in enumerate(edge_categories_lists):
            for j, _ in enumerate(edge_categories_lists[i]):
                #invert category index to make 0 <=> no category
                edge_categories[i][j] = max_category - edge_categories_lists[i][j]    

    if args.property_column:
        properties = data[args.property_column][keys]

    del data

    train_idx, test_idx = map(np.array, train_test_split(structures.index, test_size=0.20))

    charset = list(reduce(lambda x, y: set(y) | x, structures, set()))
    charset.sort()

    one_hot_encoded_fn = lambda row: map(lambda x: one_hot_array(x, len(charset)),
                                         one_hot_index(row, charset))

    charset_cats = list(reduce(lambda x, y: set(y) | x, edge_categories, set()))
    charset_cats.sort()

    one_hot_encoded_cats_fn = lambda row: map(lambda x: one_hot_array(x, len(charset_cats)),
                                              one_hot_index(row, charset_cats))

    h5f = h5py.File(args.outfile, 'w')
    h5f.create_dataset('charset', data=charset)
    h5f.create_dataset('charset_cats', data=charset_cats)

    def create_chunk_dataset(h5file, dataset_name, dataset, dataset_shape,
                             chunk_size=200, apply_fn=None):
        new_data = h5file.create_dataset(dataset_name, dataset_shape,
                                         chunks=tuple([chunk_size]+list(dataset_shape[1:])))
        for (chunk_ixs, chunk) in chunk_iterator(dataset):
            if not apply_fn:
                new_data[chunk_ixs, ...] = chunk
            else:
                new_data[chunk_ixs, ...] = apply_fn(chunk)

    create_chunk_dataset(h5f, 'data_train', train_idx,
                         (len(train_idx), MAX_WORD_LENGTH, len(charset)),
                         apply_fn=lambda ch: np.array(map(one_hot_encoded_fn,
                                                          structures[ch])))
    create_chunk_dataset(h5f, 'data_test', test_idx,
                         (len(test_idx), MAX_WORD_LENGTH, len(charset)),
                         apply_fn=lambda ch: np.array(map(one_hot_encoded_fn,
                                                          structures[ch])))

    if edge_categories.shape[0] > 0:
        create_chunk_dataset(h5f, 'categories_train', train_idx,
                             (len(train_idx), MAX_WORD_LENGTH, len(charset_cats)),
                             apply_fn=lambda c: np.array(map(one_hot_encoded_cats_fn, edge_categories[c].tolist())))

        create_chunk_dataset(h5f, 'categories_test', test_idx,
                             (len(test_idx), MAX_WORD_LENGTH, len(charset_cats)),
                             apply_fn=lambda c: np.array(map(one_hot_encoded_cats_fn, edge_categories[c].tolist())))

    if args.property_column:
        h5f.create_dataset('property_train', data=properties[train_idx])
        h5f.create_dataset('property_test', data=properties[test_idx])
    h5f.close()

if __name__ == '__main__':
    main()
