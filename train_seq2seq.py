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

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint

from neuralnets.utils import load_categories_dataset, decode_smiles_from_indexes
from keras.utils import plot_model
import matplotlib.pyplot as plt

#input_token_index = dict(
#    [(char, i) for i, char in enumerate(input_characters)])
#target_token_index = dict(
#    [(char, i) for i, char in enumerate(target_characters)])

#encoder_input_data = np.zeros(
#    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
#    dtype='float32')
#decoder_input_data = np.zeros(
#    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
#    dtype='float32')
#decoder_target_data = np.zeros(
#    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
#    dtype='float32')

#for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
#    for t, char in enumerate(input_text):
#        encoder_input_data[i, t, input_token_index[char]] = 1.
#    for t, char in enumerate(target_text):
#        # decoder_target_data is ahead of decoder_input_data by one timestep
#        decoder_input_data[i, t, target_token_index[char]] = 1.
#        if t > 0:
#            # decoder_target_data will be ahead by one timestep
#            # and will not include the start character.
#            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

class Seq2SeqAE():

    autoencoder = None

    def create(self,
               input_charset,
               output_charset,
               max_length=120,
               latent_dim=292,
               weights_file = None):

        num_encoder_tokens = len(input_charset)
        num_decoder_tokens = len(output_charset)

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder_lstm = LSTM(latent_dim, name='enc_lstm', return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, name='dec_lstm', return_sequences=True, return_state=True)
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,
                                                initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, name='dec_dense', activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.autoencoder = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Define sampling models
        self.encoder = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, name='dec_lstm', initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)

        self.autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, input_charset, output_charset, weights_file, latent_dim=292):
        self.create(input_charset, output_charset, weights_file=weights_file, latent_dim=latent_dim)

    def decode_sequence(self,
                        input_seq,
                        output_charset,
                        max_length=120):
        num_decoder_tokens = len(output_charset)
        max_category = max(output_charset)

        # Encode the input as state vectors.
        states_value = self.encoder.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        #target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sequence = [0]
        while not stop_condition:
            output_tokens, h, c = self.decoder.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_category = max_category - output_charset[sampled_token_index]
            decoded_sequence.append(sampled_category)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_category == 0 or
               len(decoded_sequence) > max_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sequence



NUM_EPOCHS = 1
BATCH_SIZE = 200
LATENT_DIM = 292
WORD_LENGTH = 120

def get_arguments():
    parser = argparse.ArgumentParser(description='Sequence to sequence autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--word_length', type=int, metavar='N', default=WORD_LENGTH,
                        help='Length of input sequences')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    return parser.parse_args()

def main():
    args = get_arguments()
    data_train, categories_train, data_test, categories_test, charset, charset_cats = load_categories_dataset(args.data)

    num_encoder_tokens = len(charset)
    num_decoder_tokens = len(charset_cats)

    print('Number of samples: ', data_train.shape[0])
    print('Number of unique input tokens: ', num_encoder_tokens)
    print('Number of unique output tokens: ', num_decoder_tokens)
    print('Max sequence length (input, output): ', (data_train.shape[1], categories_train.shape[1]))

    encoder_input_data = data_train
    decoder_input_data = categories_train
    decoder_target_data = categories_train
    #shift decoder_target_data one character to the left
    for char_id in range(1, decoder_target_data.shape[1]):
        tmp = decoder_target_data[:,char_id - 1, :]
        decoder_target_data[:,char_id - 1, :] = decoder_target_data[:,char_id, :]
        decoder_target_data[:,char_id, :] = tmp

    model = Seq2SeqAE()
    if os.path.isfile(args.model):
        model.load(charset, charset_cats, args.model, latent_dim=args.latent_dim)
    else:
        model.create(charset, charset_cats, latent_dim=args.latent_dim)

    checkpointer = ModelCheckpoint(filepath=args.model,
                                   verbose=1,
                                   save_best_only=True)

    filename, ext = os.path.splitext(args.model)
    plot_model(model.autoencoder, to_file=filename + '_nn.pdf', show_shapes=True)

    history = model.autoencoder.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                          batch_size=args.batch_size,
                          epochs=args.epochs,
                          validation_split=0.2,
                          callbacks=[checkpointer])

    # Save model
    model.autoencoder.save(args.model)

    # summarize history for loss
    plt.plot(history.history['categorical_crossentropy'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(filename + '_loss_history.pdf', bbox_inches='tight')

    for word_id, word in enumerate(data_test):
        decoded_seq = model.decode_sequence(word, charset_cats)
        print '--------------------------------'
        print 'test string: ', decode_smiles_from_indexes(word)
        print 'test categories   :', categories_test[word_id]
        print 'decoded categories:', decoded_seq
        if word_id > 10:
            break

    # # Next: inference mode (sampling).
    # # Here's the drill:
    # # 1) encode input and retrieve initial decoder state
    # # 2) run one step of decoder with this initial state
    # # and a "start of sequence" token as target.
    # # Output will be the next target token
    # # 3) Repeat with the current target token and current states

    # # Define sampling models
    # encoder_model = Model(encoder_inputs, encoder_states)

    # decoder_state_input_h = Input(shape=(args.latent_dim,))
    # decoder_state_input_c = Input(shape=(args.latent_dim,))
    # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    # decoder_outputs, state_h, state_c = decoder_lstm(
    #     decoder_inputs, initial_state=decoder_states_inputs)
    # decoder_states = [state_h, state_c]
    # decoder_outputs = decoder_dense(decoder_outputs)
    # decoder_model = Model(
    #     [decoder_inputs] + decoder_states_inputs,
    #     [decoder_outputs] + decoder_states)

    # def decode_sequence(input_seq):
    #     # Encode the input as state vectors.
    #     states_value = encoder_model.predict(input_seq)

    #     # Generate empty target sequence of length 1.
    #     target_seq = np.zeros((1, 1, num_decoder_tokens))
    #     # Populate the first character of target sequence with the start character.
    #     target_seq[0, 0, target_token_index['\t']] = 1.

    #     # Sampling loop for a batch of sequences
    #     # (to simplify, here we assume a batch of size 1).
    #     stop_condition = False
    #     decoded_sentence = ''
    #     while not stop_condition:
    #         output_tokens, h, c = decoder_model.predict(
    #             [target_seq] + states_value)

    #         # Sample a token
    #         sampled_token_index = np.argmax(output_tokens[0, -1, :])
    #         sampled_char = reverse_target_char_index[sampled_token_index]
    #         decoded_sentence += sampled_char

    #         # Exit condition: either hit max length
    #         # or find stop character.
    #         if (sampled_char == '\n' or
    #            len(decoded_sentence) > max_decoder_seq_length):
    #             stop_condition = True

    #         # Update the target sequence (of length 1).
    #         target_seq = np.zeros((1, 1, num_decoder_tokens))
    #         target_seq[0, 0, sampled_token_index] = 1.

    #         # Update states
    #         states_value = [h, c]

    #     return decoded_sentence

    # # Reverse-lookup token index to decode sequences back to
    # # something readable.
    # reverse_input_char_index = dict(
    #     (i, char) for char, i in input_token_index.items())
    # reverse_target_char_index = dict(
    #     (i, char) for char, i in target_token_index.items())

    # for seq_index in range(100):
    #     # Take one sequence (part of the training set)
    #     # for trying out decoding.
    #     input_seq = encoder_input_data[seq_index: seq_index + 1]
    #     decoded_sentence = decode_sequence(input_seq)
    #     print('-')
    #     print('Input sentence:', input_texts[seq_index])
    #     print('Decoded sentence:', decoded_sentence)

if __name__ == '__main__':
    main()
