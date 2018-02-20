'''Sequence to sequence example in Keras (character-level).
This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

# Summary of the algorithm

- We start with input sequences from a domain (e.g. English sentences)
    and correspding target sequences from another domain
    (e.g. French sentences).

- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).

- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.

- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.

# Data download

English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip

Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/

# References

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215

- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078

'''

from __future__ import print_function
import argparse
import os
import numpy as np

from neuralnets.seq2seq import Seq2SeqAE
from neuralnets.grammar import TilingGrammar
from neuralnets.utils import load_categories_dataset, decode_smiles_from_indexes, from_one_hot_array

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model

import matplotlib.pyplot as plt

NUM_EPOCHS = 1
BATCH_SIZE = 200
LATENT_DIM = 292
WORD_LENGTH = 120

def get_arguments():
    parser = argparse.ArgumentParser(description='Sequence to sequence autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('grammar', type=str, help='The HDF5 file with the tiling grammar.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--word_length', type=int, metavar='N', default=WORD_LENGTH,
                        help='Length of input sequences')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    return parser.parse_args()

def decode_sequence(model,
                    input_seq,
                    input_mask,
                    input_len,
                    output_charset,
                    bounds=None,
                    max_length=120):
    num_decoder_tokens = len(output_charset)
    max_category = max(output_charset)

    # Encode the input as state vectors.
    #states_value = model.encoder.predict(input_seq)
    states_value = model.encoder.predict([input_seq, input_mask])#mask

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_mask = np.zeros((1, 1, num_decoder_tokens))#mask

    # Populate the first character of target sequence with the start character.
    #target_seq[0, 0, max_category] = 1.
    target_min_bound = np.zeros(input_len)
    target_max_bound = np.full(input_len, -1)

    if bounds != None:
        target_min_bound = np.array([pair[0] for pair in bounds]) 
        target_max_bound = np.array([pair[1] for pair in bounds]) 

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sequence = []
    while not stop_condition:
        #Update the target mask
        char_id = len(decoded_sequence)
        for t_id in range(num_decoder_tokens):
            if input_mask[0][char_id][t_id] > 0.:
                target_mask[0][0][t_id] = 1.
            else:
                target_mask[0][0][t_id] = 0.

        output_tokens, h, c = model.decoder.predict(
            [target_seq, target_mask] + states_value)

        min_bound = target_min_bound[char_id]
        max_bound = target_max_bound[char_id]
        # if bounds != None:
        #     min_bound = max_category - target_max_bound[char_id] + 1
        #     max_bound = max_category - target_min_bound[char_id] + 1
        
        # Sample a token
        sampled_token_index = num_decoder_tokens - 1
        if min_bound < max_bound or max_bound == -1:
            sampled_token_index = min_bound + np.argmax(output_tokens[0, -1, min_bound:max_bound])
            sampled_category = output_charset[sampled_token_index]
            decoded_sequence.append(sampled_category)
        else:
            decoded_sequence.append(max_category)

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_sequence) >= input_len:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sequence

def main():
    args = get_arguments()

    tile_grammar = TilingGrammar([])
    tile_grammar.load(args.grammar)

    data_train, categories_train, masks_train, data_test, categories_test, masks_test, charset, charset_cats = load_categories_dataset(args.data)

    num_encoder_tokens = len(charset)
    num_decoder_tokens = len(charset_cats)
    #max_category = max(charset_cats)

    if categories_train.shape != masks_train.shape or data_train.shape[0] != categories_train.shape[0] or data_train.shape[1] != categories_train.shape[1]:
        print('Incompatible input array dimensions')
        print('Sample categories shape: ', categories_train.shape)
        print('Sample masks shape: ', masks_train.shape)
        print('Sample data shape: ', data_train.shape)

    print('Number of unique input tokens: ', num_encoder_tokens)
    print('Number of unique output tokens: ', num_decoder_tokens)

    encoder_input_data = np.zeros(data_train.shape, dtype='float32')
    decoder_input_data = np.zeros(categories_train.shape, dtype='float32')
    decoder_input_masks = np.zeros(categories_train.shape, dtype='float32')
    decoder_target_data = np.zeros(categories_train.shape, dtype='float32')
    num_wrong_masks = 0
    for w_id in range(encoder_input_data.shape[0]):
        for c_id in range(encoder_input_data.shape[1]):
            for one_h_id in range(encoder_input_data.shape[2]):
                if data_train[w_id][c_id][one_h_id] > 0:
                    encoder_input_data[w_id][c_id][one_h_id] = 1.
            for one_h_id_c in range(decoder_input_data.shape[2]):
                if categories_train[w_id][c_id][one_h_id_c] > 0:
                    decoder_input_data[w_id][c_id][one_h_id_c] = 1.
                    if c_id > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        decoder_target_data[w_id][c_id-1][one_h_id_c] = 1.
                if masks_train[w_id][c_id][one_h_id_c] > 0:
                    decoder_input_masks[w_id][c_id][one_h_id_c] = 1.
    if num_wrong_masks > 0:
        print('Found ' + str(num_wrong_masks) + ' wrong masks')

    encoder_test_data =  np.zeros(data_test.shape, dtype='float32')
    decoder_test_masks = np.zeros(categories_test.shape, dtype='float32')
    for w_id in range(data_test.shape[0]):
        for c_id in range(data_test.shape[1]):
            for one_h_id in range(data_test.shape[2]):
                if data_test[w_id][c_id][one_h_id] > 0:
                    encoder_test_data[w_id][c_id][one_h_id] = 1.
            for one_h_id_c in range(categories_test.shape[2]):
                if masks_test[w_id][c_id][one_h_id_c] > 0:
                    decoder_test_masks[w_id][c_id][one_h_id_c] = 1.

    model = Seq2SeqAE()
    if os.path.isfile(args.model):
        model.load(charset, charset_cats, args.model, latent_dim=args.latent_dim)
    else:
        model.create(charset, charset_cats, latent_dim=args.latent_dim)

    checkpointer = ModelCheckpoint(filepath=args.model,
                                   verbose=1,
                                   save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                    factor = 0.2,
                                    patience = 3,
                                    min_lr = 0.0001)

    filename, ext = os.path.splitext(args.model)
    plot_model(model.autoencoder, to_file=filename + '_nn.pdf', show_shapes=True)

    history = model.autoencoder.fit([encoder_input_data, decoder_input_data, decoder_input_masks], decoder_target_data,
                          batch_size=args.batch_size,
                          epochs=args.epochs,
                          validation_split=0.2,
                          callbacks=[checkpointer, reduce_lr])

    # Save model
    model.autoencoder.save(args.model)

    # summarize history for loss
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(ymin=0, ymax=2.0)
    plt.savefig(filename + '_loss_history.pdf', bbox_inches='tight')

    #test-decode a couple of train examples
    sample_ids = np.random.randint(0, len(data_train), 4)
    for word_id in sample_ids:
        print ('===============================')
        train_string = decode_smiles_from_indexes(map(from_one_hot_array, data_train[word_id]), charset)
        print ('train string: ', train_string)
        input_seq = encoder_input_data[word_id: word_id + 1]
        input_mask = decoder_input_masks[word_id: word_id + 1]
        category_bounds = tile_grammar.smiles_to_categories_bounds(train_string)
        decoded_seq = decode_sequence(model, input_seq, input_mask, len(train_string), charset_cats, category_bounds)
        train_sequence = []
        for char_id in range(categories_train[word_id].shape[0]):
            token_index = np.argmax(categories_train[word_id][char_id, :])
            train_category = charset_cats[token_index]
            train_sequence.append(train_category)

        print ('train categories   :', train_sequence[:len(train_string)])
        print ('decoded categories:', decoded_seq)
        # print ('categories bounds:', tile_grammar.smiles_to_categories_bounds(train_string))


    #test-decode a couple of test examples
    sample_ids = np.random.randint(0, len(data_test), 8)
    for word_id in sample_ids:
        print ('===============================')
        test_string = decode_smiles_from_indexes(map(from_one_hot_array, data_test[word_id]), charset)
        print ('test string: ', test_string)
        input_seq = encoder_test_data[word_id: word_id + 1]
        input_mask = decoder_test_masks[word_id: word_id + 1]
        category_bounds = tile_grammar.smiles_to_categories_bounds(test_string)
        decoded_seq = decode_sequence(model, input_seq, input_mask, len(test_string), charset_cats, category_bounds)
        test_sequence = []
        for char_id in range(categories_test[word_id].shape[0]):
            token_index = np.argmax(categories_test[word_id][char_id, :])
            test_category = charset_cats[token_index]
            test_sequence.append(test_category)

        print ('test categories   :', test_sequence[:len(test_string)])
        print ('decoded categories:', decoded_seq)
        # print ('categories bounds:', tile_grammar.smiles_to_categories_bounds(test_string))


if __name__ == '__main__':
    main()
