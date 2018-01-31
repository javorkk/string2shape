import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense

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
        encoder_lstm = LSTM(latent_dim, return_state=True, name='enc_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, num_decoder_tokens), name='dec_inputs')
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='dec_lstm')
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,
                                                initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='dec_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.autoencoder = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Define sampling models
        self.encoder = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,), name='dec_input_h')
        decoder_state_input_c = Input(shape=(latent_dim,), name='dec_input_c')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
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