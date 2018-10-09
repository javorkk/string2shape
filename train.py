from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import random

import neuralnets.grammar as grammar
from neuralnets.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset

from neuralnets.autoencoder import TilingVAE, Tiling_LSTM_VAE, Tiling_Triplet_LSTM_VAE, Tiling_LSTM_VAE_XL
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, CSVLogger, ProgbarLogger

from keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']

NUM_EPOCHS = 1
BATCH_SIZE = 200
LATENT_DIM = 292
TYPE = 'lstm'

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--grammar', type=str, default='', help='Tiling grammar. Activates (and necessary for) triplet learning')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--type', type=str, default=TYPE,
                        help='What type model to train: simple, lstm, lstm_large.')
    return parser.parse_args()

class PlotLearning(Callback):
 
    def set_filename(self, name='filename'):
        self.filename = name

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []
        self.metrics = []
        for k in self.params['metrics']:
            self.metrics.append([])

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.i += 1

        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        for  idx, name in  enumerate(self.params['metrics']):
            if not 'acc' in name:
                self.metrics[idx].append(logs.get(name))
                ax1.plot(self.x, self.metrics[idx], label=name)
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.savefig(self.filename + '_loss_history.pdf', bbox_inches='tight')
        plt.close()

class TriplesShuffle(Callback):
 
    def __init__(self, train=None, test=None, charset=None, tile_grammar=None):
        self.train_x = train
        self.train_y = train
        self.train_z = train

        self.test_x = test
        self.test_y = test
        self.test_z = test

        self.grammar = tile_grammar

        self.charset = charset

    def on_epoch_begin(self, epoch, logs=None):
        for item_id in range(len(self.train_x)):
            train_str = decode_smiles_from_indexes(map(from_one_hot_array, self.train_x[item_id]), self.charset)

            sample_ids = np.random.randint(len(self.train_x), size=10)
            str_samples = [decode_smiles_from_indexes(map(from_one_hot_array, self.train_x[x]), self.charset) for x in sample_ids]
            str_distances = [self.grammar.word_similarity(train_str, x) for x in str_samples]
            idx_y = str_distances.index(min(str_distances))
            idx_z = str_distances.index(max(str_distances))
            self.train_y[item_id] = self.train_x[sample_ids[idx_y]]
            self.train_z[item_id] = self.train_x[sample_ids[idx_z]]

        for item_id in range(len(self.test_x)):
            test_str = decode_smiles_from_indexes(map(from_one_hot_array, self.test_x[item_id]), self.charset)

            sample_ids = np.random.randint(len(self.test_x), size=10)
            str_samples = [decode_smiles_from_indexes(map(from_one_hot_array, self.test_x[x]), self.charset) for x in sample_ids]
            str_distances = [self.grammar.word_similarity(test_str, x) for x in str_samples]
            idx_y = str_distances.index(min(str_distances))
            idx_z = str_distances.index(max(str_distances))
            self.test_y[item_id] = self.test_x[sample_ids[idx_y]]
            self.test_z[item_id] = self.test_x[sample_ids[idx_z]]


def main():
    args = get_arguments()
    data_train, data_test, charset = load_dataset(args.data)

    word_length = data_train.shape[1]
    print("----------- max word length is ",word_length, " -----------------")

    #print ("Grammar characters: ") 
    #print (charset)
    #print("data dtype is " + str(data_train.dtype))
    #print("vector dtype is " + str(data_train[0].dtype))
    #print("vector shape is " + str(data_train[0].shape))

    #for i in range(1):
    #    sample_id = np.random.randint(0, len(data_train))
    #    exaple = data_train[sample_id]
    #    print("training vector " + str(sample_id) + ":")
    #    print(exaple)

    #return

    if os.path.isfile(args.grammar):
        model = Tiling_Triplet_LSTM_VAE()
    elif args.type == 'lstm':
        model = Tiling_LSTM_VAE()
    elif args.type == 'lstm_large':
        model = Tiling_LSTM_VAE_XL()
    elif args.type == 'simple':
        model = TilingVAE()
    else:
        model = Tiling_LSTM_VAE()

    if os.path.isfile(args.model):
        model.load(charset, args.model, max_w_length=word_length, latent_rep_size=args.latent_dim)
    else:
        model.create(charset, max_length=word_length, latent_rep_size=args.latent_dim)

    print("available metrics: ", model.autoencoder.metrics_names)

    checkpointer = ModelCheckpoint(
                                monitor='val_loss',
                                filepath = args.model,
                                verbose = 1,
                                mode = 'min',
                                save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                factor = 0.2,
                                patience = 3,
                                min_lr = 0.00000001)

    filename, ext = os.path.splitext(args.model) 
    plot_model(model.autoencoder, to_file=filename + '_nn.pdf', show_shapes=True)

    csv_logger = CSVLogger(filename + '_training.log', append=True)

    plot = PlotLearning()
    plot.set_filename(filename)

    if os.path.isfile(args.grammar):
        tiling_grammar = grammar.TilingGrammar([])
        tiling_grammar.load(args.grammar)

        tri_shuffle = TriplesShuffle(train=data_train,
                                    test=data_test,
                                    charset=charset,
                                    tile_grammar=tiling_grammar)

        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                    factor = 0.2,
                                    patience = 3,
                                    min_lr = 0.0000000001)

        history = model.autoencoder.fit(
            {'main_input':tri_shuffle.train_x, 'positive_input':tri_shuffle.train_y, 'negative_input':tri_shuffle.train_z},
            tri_shuffle.train_x,
            shuffle = True,
            epochs = args.epochs,
            batch_size = args.batch_size,
            callbacks = [tri_shuffle, checkpointer, reduce_lr, plot, csv_logger],
            validation_data = ({'main_input':tri_shuffle.test_x, 'positive_input':tri_shuffle.test_y, 'negative_input':tri_shuffle.test_z}, tri_shuffle.test_x)
        )

    else:

        history = model.autoencoder.fit(
            data_train,
            data_train,
            shuffle = True,
            epochs = args.epochs,
            batch_size = args.batch_size,
            callbacks = [checkpointer, reduce_lr, plot, csv_logger],
            validation_data = (data_test, data_test)
        )

    # summarize history for loss
    # plt.plot(history.history['val_loss'])
    # plt.title('loss breakdown')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.savefig(filename + '_losses_breakdown.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
