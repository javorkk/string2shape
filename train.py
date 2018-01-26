from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

from neuralnets.autoencoder import TilingVAE
from neuralnets.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.utils import plot_model
import matplotlib.pyplot as plt

NUM_EPOCHS = 1
BATCH_SIZE = 200
LATENT_DIM = 292

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    return parser.parse_args()

def main():
    args = get_arguments()
    data_train, data_test, charset = load_dataset(args.data)

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

    model = TilingVAE()
    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = args.latent_dim)
    else:
        model.create(charset, latent_rep_size = args.latent_dim)

    checkpointer = ModelCheckpoint(filepath = args.model,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    filename, ext = os.path.splitext(args.model) 
    plot_model(model.autoencoder, to_file=filename + '_nn.pdf', show_shapes=True)

    history = model.autoencoder.fit(
        data_train,
        data_train,
        shuffle = True,
        epochs = args.epochs,
        batch_size = args.batch_size,
        callbacks = [checkpointer, reduce_lr],
        validation_data = (data_test, data_test)
    )

    # summarize history for loss
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(filename + '_loss_history.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
