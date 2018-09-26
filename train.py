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
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, CSVLogger

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

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="validation loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.savefig(self.filename + '_loss_history.pdf', bbox_inches='tight')
        plt.close()

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

    if args.type == 'lstm':
        model = Tiling_LSTM_VAE()
    elif args.type == 'lstm_large':
        model = Tiling_LSTM_VAE_XL()
    elif args.type == 'simple':
        model = TilingVAE()
    elif os.path.isfile(args.grammar):
        model = Tiling_Triplet_LSTM_VAE()
    else:
        model = Tiling_LSTM_VAE()

    if os.path.isfile(args.model):
        model.load(charset, args.model, max_w_length=word_length, latent_rep_size=args.latent_dim)
    else:
        model.create(charset, max_length=word_length, latent_rep_size=args.latent_dim)

    checkpointer = ModelCheckpoint(filepath = args.model,
                                verbose = 1,
                                save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                factor = 0.2,
                                patience = 3,
                                min_lr = 0.000001)

    filename, ext = os.path.splitext(args.model) 
    plot_model(model.autoencoder, to_file=filename + '_nn.pdf', show_shapes=True)

    csv_logger = CSVLogger(filename + '_training.log')

    plot = PlotLearning()
    plot.set_filename(filename)

    if os.path.isfile(args.grammar):
        tiling_grammar = grammar.TilingGrammar([])
        tiling_grammar.load(args.grammar)
        data_y = data_train
        data_z = data_train

        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                    factor = 0.2,
                                    patience = 3,
                                    min_lr = 0.000001)

        for item_id in range(len(data_train)):
            train_str = decode_smiles_from_indexes(map(from_one_hot_array, data_train[item_id]), charset)

            sample_ids = np.random.randint(len(data_train), size=100)
            str_samples = [decode_smiles_from_indexes(map(from_one_hot_array, data_train[x]), charset) for x in sample_ids]
            str_distances = [tiling_grammar.word_similarity(train_str, x) for x in str_samples]
            idx_y = str_distances.index(min(str_distances))
            idx_z = str_distances.index(max(str_distances))
            data_y[item_id] = data_train[sample_ids[idx_y]]
            data_z[item_id] = data_train[sample_ids[idx_z]]

        test_y = data_test
        test_z = data_test
        for item_id in range(len(data_test)):
            test_str = decode_smiles_from_indexes(map(from_one_hot_array, data_test[item_id]), charset)

            sample_ids = np.random.randint(len(data_test), size=100)
            str_samples = [decode_smiles_from_indexes(map(from_one_hot_array, data_test[x]), charset) for x in sample_ids]
            str_distances = [tiling_grammar.word_similarity(test_str, x) for x in str_samples]
            idx_y = str_distances.index(min(str_distances))
            idx_z = str_distances.index(max(str_distances))
            test_y[item_id] = data_test[sample_ids[idx_y]]
            test_z[item_id] = data_test[sample_ids[idx_z]]

        history = model.autoencoder.fit(
            {'main_input':data_train, 'positive_input':data_y, 'negative_input':data_z},
            data_train,
            shuffle = True,
            epochs = args.epochs,
            batch_size = args.batch_size,
            callbacks = [checkpointer, reduce_lr, plot, csv_logger],
            validation_data = ({'main_input':data_test, 'positive_input':test_y, 'negative_input':test_z}, data_test)
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
