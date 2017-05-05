from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

from neuralnets.graph_autoencoder import GraphVAE
from neuralnets.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_graph_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, ProgbarLogger

from keras.utils import visualize_util
import matplotlib.pyplot as plt

import keras.backend as K

NUM_EPOCHS = 1
BATCH_SIZE = 600
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
    data_train, data_test, charset, connectivity_dims = load_graph_dataset(args.data)

    
    #print ("Max node degree: " + str(connectivity_dims))
    #print ("Grammar characters: ") 
    #print (charset)
    #print("data dtype is " + str(data_train.dtype))
    #print("vector dtype is " + str(data_train[0].dtype))
    #print("vector shape is " + str(data_train[0].shape))

    #conn_dim_start  = data_train[0].shape[1] - connectivity_dims

    #print("conn_dim_start is " + str(conn_dim_start))

    #def t_loss(x_true, x_pred):
    #    max_length_f = 120.0
    #    x_true_type = x_true[:,:conn_dim_start]
    #    x_pred_type = x_pred[:,:conn_dim_start]

    #    x_true_type = K.flatten(x_true_type)
    #    x_pred_type = K.flatten(x_pred_type)

    #    return max_length_f * objectives.binary_crossentropy(x_true_type, x_pred_type)

    #def c_loss(x_true, x_pred):
    #    max_length_f = 120.0
    #    x_true_conn = x_true[:,conn_dim_start:]
    #    x_pred_conn = x_pred[:,conn_dim_start:]

    #    #x_true_conn = 0.5 * x_true_conn + 0.5
    #    x_pred_conn = 0.5 * K.round(x_pred_conn * max_length_f) / max_length_f + 0.5
    #    x_pred_conn = 0.5 * x_pred_conn + 0.5
    #    x_pred_conn.sort(axis=1)
    #    x_true_conn.sort(axis=1)
    #    x_true_conn = K.flatten(x_true_conn)
    #    x_pred_conn = K.flatten(x_pred_conn)

    #    return max_length_f * objectives.mean_absolute_error(x_true_conn, x_pred_conn)

    #def vae_loss(x_true, x_pred):    
    #    return 0.5 * t_loss(x_true, x_pred) + 0.25 * c_loss(x_true, x_pred)
        
    #def type_acc(x_true, x_pred):
    #    y_true = x_true[:,:conn_dim_start]
    #    y_pred = x_pred[:,:conn_dim_start]
    #    y_true = K.flatten(y_true)
    #    y_pred = K.flatten(y_pred)
    #    return   K.mean(K.cast(K.equal(y_true, K.round(y_pred)), K.floatx()), axis=-1)

    #vec_width = connectivity_dims + len(charset)
    #for i in range(1):
    #    sample_id = np.random.randint(0, len(data_train))
    #    exaple = data_train[sample_id]
    #    print("training vector " + str(sample_id) + " \n type:")
    #    print(exaple[:,:conn_dim_start])
    #    print("connectivity:")
    #    print(exaple[:,conn_dim_start:])

    #    #exaple_0 = data_train[sample_id]
    #    #exaple_1 = data_train[sample_id]
    #    #print("t_loss self: " + str(t_loss(exaple_0.reshape(1, 120,vec_width), exaple_1.reshape(1, 120,vec_width))))
    #    #print("c_loss self: " + str(c_loss(exaple_0.reshape(1, 120,vec_width), exaple_1.reshape(1, 120,vec_width))))
    #    #print("vae_loss self: " + str(vae_loss(exaple_0.reshape(1, 120,vec_width), exaple_1.reshape(1, 120,vec_width))))
    #    #print("type_acc self: " + str(type_acc(exaple_0.reshape(1, 120,vec_width), exaple_1.reshape(1, 120,vec_width))))

    #return

    model = GraphVAE()
    if os.path.isfile(args.model):
        model.load(charset, args.model, connectivity_dims, latent_rep_size = args.latent_dim)
    else:
        model.create(charset, connectivity_dims, latent_rep_size = args.latent_dim)

    checkpointer = ModelCheckpoint(filepath = args.model,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    filename, ext = os.path.splitext(args.model) 
    visualize_util.plot(model.autoencoder, to_file=filename + '_nn.pdf', show_shapes=True)

    history = model.autoencoder.fit(
        data_train,
        data_train,
        shuffle = True,
        nb_epoch = args.epochs,
        batch_size = args.batch_size,
        callbacks = [checkpointer, reduce_lr],
        validation_data = (data_test, data_test)
    )

    # summarize history for loss
    plt.plot(history.history['val_t_loss'])
    plt.plot(history.history['val_c_loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['type', 'connectivity', 'combined'], loc='upper right')
    plt.savefig(filename + '_loss_history.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
