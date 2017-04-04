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
    parser.add_argument('--sample', type=int, metavar='N', default=NUM_SAMPLES,
                        help='Number of items to sample from data generator.')
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

    sampled = model.autoencoder.predict(data[0].reshape(1, 120, len(charset))).argmax(axis=2)[0]
    mol = decode_smiles_from_indexes(map(from_one_hot_array, data[0]), charset)
    sampled = decode_smiles_from_indexes(sampled, charset)
    print(mol)
    print(sampled)

def decoder(args, model):
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
    
    for i in range(args.sample):
        decoded_data = model.decoder.predict(data[i].reshape(1, latent_dim)).argmax(axis=2)[0]
        char_data = decode_smiles_from_indexes(decoded_data, charset)

        for step_size in [0.0001, 0.001, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]:
            z_sample = np.array([np.random.random(latent_dim)]) * step_size
            if(i < len(data)):
                z_sample += data[i]
            decoded_sample = model.decoder.predict(z_sample.reshape(1, latent_dim)).argmax(axis=2)[0]
            char_sample = decode_smiles_from_indexes(decoded_sample, charset)
            if(char_sample != char_data and tiling_grammar.check_word(char_sample)):
                print("data point  : " + char_data)
                print("offset point: " + char_sample)
                print("offset magnitude: " + str(step_size))
                print("-----------------------------------------------------------------------")
                break

        #z_sample = np.ones([latent_dim])
        #for j in range(latent_dim):
        #    z_sample[j] = np.random.normal()
        #decoded_rnd_sample = model.decoder.predict(z_sample.reshape(1, latent_dim)).argmax(axis=2)[0]
        #char_rnd_sample = decode_smiles_from_indexes(decoded_rnd_sample, charset)
        #if tiling_grammar.check_word(char_rnd_sample):
        #    print("random point: " + char_rnd_sample)
        #    print("-----------------------------------------------------------------------")
    
    #for latent_vec in data:
    #    sampled = model.decoder.predict(latent_vec.reshape(1, latent_dim)).argmax(axis=2)[0]
    #    sampled = decode_smiles_from_indexes(sampled, charset)
    #    print(sampled)

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

if __name__ == '__main__':
    main()
