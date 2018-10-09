import copy
from keras import optimizers
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda, LSTM, Concatenate, LeakyReLU
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

class TilingVAE():

    autoencoder = None
    
    def create(self,
               charset,
               max_length = 120,
               latent_rep_size = 292,
               weights_file = None):
        charset_length = len(charset)
        
        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, z)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        x1 = Input(shape=(max_length, charset_length))
        vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        self.autoencoder = Model(
            x1,
            self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)

        self.autoencoder.compile(optimizer = 'Adam',
                                 loss = vae_loss,
                                 metrics = ['accuracy'])

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.125 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(701, return_sequences = True, name='out_gru_1')(h)
        h = GRU(701, return_sequences = True, name='out_gru_2')(h)
        h = GRU(701, return_sequences = True, name='out_gru_3')(h)
        return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, max_w_length=120, latent_rep_size=292):
        self.create(charset, weights_file=weights_file,  max_length=max_w_length, latent_rep_size=latent_rep_size)

class Tiling_LSTM_VAE():

    autoencoder = None
    
    def create(self,
               charset,
               max_length = 120,
               latent_rep_size = 292,
               weights_file = None):
        charset_length = len(charset)
        
        x = Input(shape=(max_length, charset_length), name='main_input')

        _, latent_x = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, latent_x)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        ae_input = Input(shape=(max_length, charset_length), name='main_input')
        vae_loss,  ae_latent_z = self._buildEncoder(ae_input, latent_rep_size, max_length)
        self.autoencoder = Model(
            ae_input,
            self._buildDecoder(
                ae_latent_z,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)

        opt = optimizers.Adam(lr=0.00001, amsgrad=True)
        self.autoencoder.compile(optimizer = opt,
                                 loss = vae_loss,
                                 metrics = ['accuracy'])

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
        h = LSTM(301, return_sequences = True, name='in_lstm_1')(x)
        h = LSTM(301, return_sequences = True, name='in_lstm_2')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.125 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = LSTM(501, return_sequences=True, name='out_lstm_1')(h)
        h = LSTM(501, return_sequences=True, name='out_lstm_2')(h)
        return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, max_w_length=120, latent_rep_size=292):
        self.create(charset, weights_file=weights_file,  max_length=max_w_length, latent_rep_size=latent_rep_size)

class Tiling_Triplet_LSTM_VAE():

    autoencoder = None
    
    def create(self,
               charset,
               max_length = 120,
               latent_rep_size = 292,
               weights_file = None):
        charset_length = len(charset)
        
        x = Input(shape=(max_length, charset_length), name='main_input')
        y = Input(shape=(max_length, charset_length), name='positive_input')
        z = Input(shape=(max_length, charset_length), name='negative_input')

        _, latent_x = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, latent_x)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        _, latent_y = self._buildEncoder(y, latent_rep_size, max_length)
        _, latent_z = self._buildEncoder(z, latent_rep_size, max_length)

        vae_loss,  latent_x = self._buildEncoder(x, latent_rep_size, max_length)
        # self.autoencoder = Model(
        #     x,
        #     self._buildDecoder(
        #         latent_x,
        #         latent_rep_size,
        #         max_length,
        #         charset_length
        #     )
        # )

        self.autoencoder = Model(
            (x,y,z),
            self._buildDecoder(
                latent_x,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        # contrastive loss on sampled latent point coordinates
        def triplet_loss(x_in, x_pred):
            lat_x = K.flatten(latent_x)
            pos_y = K.flatten(latent_y)
            neg_z = K.flatten(latent_z)
            d_pos = lat_x - pos_y
            d_neg = lat_x - neg_z
            tr_loss = K.maximum(0.0, 0.001 + K.sum(d_pos * d_pos) - K.sum(d_neg * d_neg))
            return tr_loss

        def combined_loss(x_in, x_pred):
            return vae_loss(x_in, x_pred) + triplet_loss(x_in, x_pred)

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)

        opt = optimizers.Adam(lr=0.00001, amsgrad=True)
        self.autoencoder.compile(optimizer = opt,
                                 loss = combined_loss,
                                 metrics = [triplet_loss, vae_loss, 'accuracy'])

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
        #x,y,z = input_triple
        h = LSTM(301, return_sequences = True, name='in_lstm_1')(x)
        h = LSTM(301, return_sequences = True, name='in_lstm_2')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1')(h)

        z_mean_x = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var_x = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        latent_x = Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean_x, z_log_var_x])

        # h_y = LSTM(301, return_sequences = True, name='in_lstm_1')(y)
        # h_y = LSTM(301, return_sequences = True, name='in_lstm_2')(h_y)
        # h_y = Flatten(name='flatten_1')(h_y)
        # h_y = Dense(435, activation = 'relu', name='dense_1')(h_y)

        # z_mean_y = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h_y)
        # z_log_var_y = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h_y)

        # latent_y = Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean_y, z_log_var_y])

        # h_z = LSTM(301, return_sequences = True, name='in_lstm_1')(z)
        # h_z = LSTM(301, return_sequences = True, name='in_lstm_2')(h_z)
        # h_z = Flatten(name='flatten_1')(h_z)
        # h_z = Dense(435, activation = 'relu', name='dense_1')(h_z)

        # z_mean_z = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h_z)
        # z_log_var_z = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h_z)

        # latent_z = Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean_z, z_log_var_z])

        def vae_loss(x_in, x_decoded):
            #x,y,z = x_in
            x = K.flatten(x_in)
            x_decoded = K.flatten(x_decoded)
            #cross entropy
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded)
            #Kullback-Leibler regularization
            kl_loss = - 0.25 * K.sum(1 + z_log_var_x - K.square(z_mean_x) - K.exp(z_log_var_x), axis = -1)
            return xent_loss + kl_loss
            #triplet contrastive loss on Euclidean distance in latent space
            # flat_x = K.flatten(z_mean_x)
            # flat_y = K.flatten(z_mean_y)
            # flat_z = K.flatten(z_mean_z)
            # d_pos = flat_x - flat_y
            # d_neg = flat_x - flat_z
            # tr_loss = K.max(0.0, 1.0 + K.dot(d_pos, d_pos) - K.dot(d_neg, d_neg))
            #return xent_loss + kl_loss + 0.125 * tr_loss

        return (vae_loss, latent_x)

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = LSTM(501, return_sequences=True, name='out_lstm_1')(h)
        h = LSTM(501, return_sequences=True, name='out_lstm_2')(h)
        return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, max_w_length=120, latent_rep_size=292):
        self.create(charset, weights_file=weights_file,  max_length=max_w_length, latent_rep_size=latent_rep_size)

class Tiling_LSTM_VAE_XL():

    autoencoder = None

    def create(self,
               charset,
               max_length = 120,
               latent_rep_size = 292,
               weights_file = None):
        charset_length = len(charset)

        x = Input(shape=(max_length, charset_length), name='main_input')
        y = Input(shape=(max_length, charset_length), name='positive_input')
        z = Input(shape=(max_length, charset_length), name='negative_input')

        _, latent_x = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, latent_x)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        _, latent_y = self._buildEncoder(y, latent_rep_size, max_length)
        _, latent_z = self._buildEncoder(z, latent_rep_size, max_length)

        vae_loss,  latent_x = self._buildEncoder(x, latent_rep_size, max_length)
        # self.autoencoder = Model(
        #     x,
        #     self._buildDecoder(
        #         latent_x,
        #         latent_rep_size,
        #         max_length,
        #         charset_length
        #     )
        # )

        self.autoencoder = Model(
            (x,y,z),
            self._buildDecoder(
                latent_x,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        # contrastive loss on sampled latent point coordinates
        def triplet_loss(x_in, x_pred):
            lat_x = K.flatten(latent_x)
            pos_y = K.flatten(latent_y)
            neg_z = K.flatten(latent_z)
            d_pos = lat_x - pos_y
            d_neg = lat_x - neg_z
            tr_loss = K.maximum(0.0, 1.0 + K.sum(d_pos * d_pos) - K.sum(d_neg * d_neg))
            return tr_loss

        def combined_loss(x_in, x_pred):
            return vae_loss(x_in, x_pred) + 0.25 * triplet_loss(x_in, x_pred)

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)

        opt = optimizers.Adam(lr=0.00001, amsgrad=True)
        self.autoencoder.compile(optimizer = opt,
                                 loss = combined_loss,
                                 metrics = [triplet_loss, vae_loss, 'accuracy'])


    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
        lstm_0_f = LSTM(512, return_sequences=True, name='lstm_0_f')(x)
        lstm_0_b = LSTM(512, return_sequences=True, name='lstm_0_b', go_backwards=True)(x)
        x = Concatenate(name='concatenate')([lstm_0_f, lstm_0_b])
        h = LSTM(512, return_sequences = True, name='in_lstm_1')(x)
        h = Flatten(name='flatten_1')(h)
        h = Dense(1024, name='dense_1')(h)
        h = LeakyReLU(0.3)(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.125 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input')(z)
        h = LeakyReLU(0.3)(h)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = LSTM(512, return_sequences=True, name='out_lstm_1')(h)
        h = LSTM(512, return_sequences=True, name='out_lstm_2')(h)
        h = LSTM(512, return_sequences=True, name='out_lstm_3')(h)
        return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)

    def load(self, charset, weights_file, max_w_length=120, latent_rep_size=292):
        self.create(charset, weights_file=weights_file,  max_length=max_w_length, latent_rep_size=latent_rep_size)
