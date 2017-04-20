from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys

from neuralnets.autoencoder import TilingVAE
from neuralnets.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset
import neuralnets.grammar as grammar

LATENT_DIM = 292
TARGET = 'autoencoder'
NUM_SAMPLES = 100

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='File of latent representation tensors for decoding.')
    parser.add_argument('model', type=str, help='Trained Keras model to use.')
    parser.add_argument('grammar', type=str, help='Tiling grammar.')
    parser.add_argument('--save_h5', type=str, help='Name of a file to write HDF5 output to.')
    parser.add_argument('--target', type=str, default=TARGET,
                        help='What model to sample from: autoencoder, encoder, decoder.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--samples', type=int, metavar='N', default=NUM_SAMPLES,
                        help='Number of items to sample from the data generator.')
    parser.add_argument('--require_cycle', dest='require_cycle', action='store_true',
                        help='Only return samples if they contain cycles.')
    return parser.parse_args()

def read_latent_data(filename):
    h5f = h5py.File(filename, 'r')
    data = h5f['latent_vectors'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    return (data, charset)

def autoencoder(args, model):
    latent_dim = args.latent_dim
    data, charset = load_dataset(args.data, split = False)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    for i in range(args.samples):
        sample_id = np.random.randint(0, len(data))
        sampled = model.autoencoder.predict(data[sample_id].reshape(1, 120, len(charset))).argmax(axis=2)[0]
        mol = decode_smiles_from_indexes(map(from_one_hot_array, data[sample_id]), charset)
        if args.require_cycle and mol.find("0") == -1:
            continue
        sampled = decode_smiles_from_indexes(sampled, charset)
        print("input word  : " + mol)
        print("decoded word: " + sampled)

def decoder(args, model):
    latent_dim = args.latent_dim
    data, charset = read_latent_data(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)
  
    for latent_vec in data:
        sampled = model.decoder.predict(latent_vec.reshape(1, latent_dim)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset)
        print(sampled)

def decoder_nbr(args, model):
    latent_dim = args.latent_dim
    data, charset = read_latent_data(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    tiling_grammar = grammar.TilingGrammar([])
    if os.path.isfile(args.grammar):
        tiling_grammar.load(args.grammar)
    else:
        raise ValueError("Grammar file %s doesn't exist" % args.grammar)
    
    for i in range(args.samples):
        sample_id = np.random.randint(0, len(data))
        decoded_data = model.decoder.predict(data[sample_id].reshape(1, latent_dim)).argmax(axis=2)[0]
        char_data = decode_smiles_from_indexes(decoded_data, charset)
        if not tiling_grammar.check_word(char_data):
            continue
        if args.require_cycle and char_data.find("0") == -1:
            continue

        for step_size in [0.0001, 0.001, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]:
            z_sample = np.array([np.random.random(latent_dim)]) * step_size
            z_sample += data[sample_id]
            decoded_sample = model.decoder.predict(z_sample.reshape(1, latent_dim)).argmax(axis=2)[0]
            char_sample = decode_smiles_from_indexes(decoded_sample, charset)
            if(char_sample != char_data and tiling_grammar.check_word(char_sample)):
                print("data point  : " + char_data)
                print("offset point: " + char_sample)
                print("offset magnitude: " + str(step_size))
                print("-----------------------------------------------------------------------")
                break

def decoder_lerp(args, model):
    latent_dim = args.latent_dim
    data, charset = read_latent_data(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    tiling_grammar = grammar.TilingGrammar([])
    if os.path.isfile(args.grammar):
        tiling_grammar.load(args.grammar)
    else:
        raise ValueError("Grammar file %s doesn't exist" % args.grammar)

    for i in range(args.samples):
        sample_ids = np.random.randint(0, len(data), 2)
        
        decoded_data_0 = model.decoder.predict(data[sample_ids[0]].reshape(1, latent_dim)).argmax(axis=2)[0]
        char_data_0 = decode_smiles_from_indexes(decoded_data_0, charset)

        decoded_data_1 = model.decoder.predict(data[sample_ids[1]].reshape(1, latent_dim)).argmax(axis=2)[0]
        char_data_1 = decode_smiles_from_indexes(decoded_data_1, charset)
        if not (tiling_grammar.check_word(char_data_0) and tiling_grammar.check_word(char_data_1)) :
            continue

        if args.require_cycle and char_data_0.find("0") == -1 and char_data_1.find("0") == -1:
            continue

        print("-----------------------------------------------------------------------")
        print("data point 0.0: " + char_data_0)

        for k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for step_size in [0.0001, 0.001, 0.01, 0.02, 0.05, 0.075]:
                rnd_offset = np.array([np.random.random(latent_dim)]) * step_size
                z_sample = (1.0 - k) * data[sample_ids[0]] + k * data[sample_ids[1]] + rnd_offset
                decoded_sample_k = model.decoder.predict(z_sample.reshape(1, latent_dim)).argmax(axis=2)[0]
                char_sample_k = decode_smiles_from_indexes(decoded_sample_k, charset)
                if(char_sample_k != char_data_0 and char_sample_k != char_data_1 and tiling_grammar.check_word(char_sample_k)):
                    print("data point " + str(k) + ": "  + char_sample_k + " (rnd offset = " + str(step_size) + ")")
                    break
        print("data point 1.0: " + char_data_1)
        print("-----------------------------------------------------------------------")

def _gen_latent_path(data, end_pt_0, end_pt_1, waypoints = [], depth = 0):
    if depth > 4:
        return waypoints
    
    #sample_ids = np.random.randint(0, len(data), 512)
    #sample_dist = [np.linalg.norm(data[end_pt_0] - data[x]) + np.linalg.norm(data[end_pt_1] - data[x]) for x in sample_ids]
    sample_dist = [np.linalg.norm(data[end_pt_0] - data[x]) + np.linalg.norm(data[end_pt_1] - data[x]) for x in range(len(data)) if x != end_pt_1 and x != end_pt_0]

    val, idx = min((val, idx) for (idx, val) in enumerate(sample_dist))
    _gen_latent_path(data, end_pt_0, idx, waypoints, depth + 1)
    waypoints.append(idx)
    _gen_latent_path(data, idx, end_pt_1, waypoints, depth + 1)

    return waypoints
    
def decoder_path(args, model):
    latent_dim = args.latent_dim
    data, charset = read_latent_data(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    tiling_grammar = grammar.TilingGrammar([])
    if os.path.isfile(args.grammar):
        tiling_grammar.load(args.grammar)
    else:
        raise ValueError("Grammar file %s doesn't exist" % args.grammar)

    for i in range(args.samples):
        sample_ids = np.random.randint(0, len(data), 2)
        
        decoded_data_0 = model.decoder.predict(data[sample_ids[0]].reshape(1, latent_dim)).argmax(axis=2)[0]
        char_data_0 = decode_smiles_from_indexes(decoded_data_0, charset)

        decoded_data_1 = model.decoder.predict(data[sample_ids[1]].reshape(1, latent_dim)).argmax(axis=2)[0]
        char_data_1 = decode_smiles_from_indexes(decoded_data_1, charset)
        if not (tiling_grammar.check_word(char_data_0) and tiling_grammar.check_word(char_data_1)) :
            continue

        if args.require_cycle and char_data_0.find("0") == -1 and char_data_1.find("0") == -1:
            continue

        print("---------------------sample " + str(i) + "------------------------------------------")
        print("data point  0.0: " + char_data_0)
        
        path_ids = []
        path_ids.append(sample_ids[0])
        path_ids = _gen_latent_path(data, sample_ids[0], sample_ids[1], waypoints = path_ids)
        path_ids.append(sample_ids[1])
        
        for p in range(len(path_ids) - 1):                   
            decoded_data_p = model.decoder.predict(data[path_ids[p + 1]].reshape(1, latent_dim)).argmax(axis=2)[0]
            char_data_p = decode_smiles_from_indexes(decoded_data_p, charset)
            if not tiling_grammar.check_word(char_data_p) :
                continue

            for k in [0.2, 0.4, 0.6, 0.8]:
                current_distance = np.linalg.norm(data[path_ids[p]] - data[path_ids[p + 1]])
                rnd_offset = np.array([np.random.random(latent_dim)]) * 0.1 * current_distance
                z_sample = (1.0 - k) * data[path_ids[p]] + k * data[path_ids[p + 1]] + rnd_offset
                decoded_sample_k = model.decoder.predict(z_sample.reshape(1, latent_dim)).argmax(axis=2)[0]
                char_sample_k = decode_smiles_from_indexes(decoded_sample_k, charset)
                if(char_sample_k != char_data_0 and char_sample_k != char_data_1 and char_sample_k != char_data_p and tiling_grammar.check_word(char_sample_k)):
                    print("sample point " + str(k) + ": "  + char_sample_k + " (rnd offset = " + str(0.1 * current_distance) + ")")
                    break

            if p < len(path_ids) - 2 :
                print("path waypoint " + str(p + 1) + ": "  + char_data_p)

        print("data point   1.0: " + char_data_1)
        print("-----------------------------------------------------------------------")


def decoder_rnd(args, model):
    latent_dim = args.latent_dim
    data, charset = read_latent_data(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    tiling_grammar = grammar.TilingGrammar([])
    if os.path.isfile(args.grammar):
        tiling_grammar.load(args.grammar)
    else:
        raise ValueError("Grammar file %s doesn't exist" % args.grammar)
    
    for i in range(args.samples):
        z_sample = np.random.normal(size=latent_dim)
        decoded_rnd_sample = model.decoder.predict(z_sample.reshape(1, latent_dim)).argmax(axis=2)[0]
        char_rnd_sample = decode_smiles_from_indexes(decoded_rnd_sample, charset)
        if tiling_grammar.check_word(char_rnd_sample):
            print("random point: " + char_rnd_sample)
            print("-----------------------------------------------------------------------")


def encoder(args, model):
    latent_dim = args.latent_dim
    data, charset = load_dataset(args.data, split = False)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    x_latent = model.encoder.predict(data)
    if args.save_h5:
        h5f = h5py.File(args.save_h5, 'w')
        h5f.create_dataset('charset', data = charset)
        h5f.create_dataset('latent_vectors', data = x_latent)
        h5f.close()
    else:
        np.savetxt(sys.stdout, x_latent, delimiter = '\t')

def main():
    args = get_arguments()
    model = TilingVAE()

    if args.target == 'autoencoder':
        autoencoder(args, model)
    elif args.target == 'encoder':
        encoder(args, model)
    elif args.target == 'decoder':
        decoder(args, model)
    elif args.target == 'decoder_rnd':
        decoder_rnd(args, model)
    elif args.target == 'decoder_nbr':
        decoder_nbr(args, model)
    elif args.target == 'decoder_lerp':
        decoder_lerp(args, model)
    elif args.target == 'decoder_path':
        decoder_path(args, model)


if __name__ == '__main__':
    main()
