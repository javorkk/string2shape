import copy
from keras import backend as K
from keras import objectives
from keras.engine import Layer, InputSpec
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, merge
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam



class MyCropping1D(Layer):
    """Cropping layer for 1D input (e.g. temporal sequence).

    It crops along the NON-time dimension (axis 2).

    # Arguments
        cropping: tuple of int (length 2)
            How many units should be trimmed off at the beginning and end of
            the cropping dimension (axis 1).

    # Input shape
        3D tensor with shape `(samples, axis_to_crop, features)`

    # Output shape
        3D tensor with shape `(samples, cropped_axis, features)`
    """

    def __init__(self, cropping=(1, 1), **kwargs):
        super(MyCropping1D, self).__init__(**kwargs)
        self.cropping = tuple(cropping)
        if len(self.cropping) != 2:
            raise ValueError('`cropping` must be a tuple length of 2.')
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.built = True

    def get_output_shape_for(self, input_shape):
        if input_shape[2] is not None:
            length = input_shape[2] - self.cropping[0] - self.cropping[1]
        else:
            length = None
        return (input_shape[0],
                input_shape[1],
                length)

    def call(self, x, mask=None):
        if self.cropping[1] == 0:
            return x[:, :, self.cropping[0]:]
        else:
            return x[:, :, self.cropping[0]:-self.cropping[1]]

    def get_config(self):
        config = {'cropping': self.cropping}
        base_config = super(MyCropping1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GraphVAE():

    autoencoder = None
    
    def create(self,
               charset,
               connectivity_dims = 0,
               max_length = 120,
               latent_rep_size = 292,
               weights_file = None):
        #charset_length = len(charset)
        intput_width = len(charset) + connectivity_dims
        
        x = Input(shape=(max_length, intput_width))
        _1, _2, _3, _4, z = self._buildEncoder(x, latent_rep_size, max_length, len(charset))
        self.encoder = Model(x, z)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                connectivity_dims,
                intput_width
            )
        )

        x1 = Input(shape=(max_length, intput_width))
        type_acc, type_loss, conn_loss, vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length, len(charset))
        self.autoencoder = Model(
            x1,
            self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
                connectivity_dims,
                intput_width
            )
        )

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)

        #opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.autoencoder.compile(optimizer = 'adam',
                                 loss = vae_loss,
                                 metrics = [type_acc, type_loss, conn_loss])

    def _buildEncoder(self, x, latent_rep_size, max_length, conn_dim_start, epsilon_std = 0.01):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., std = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        #def my_conn_loss(y_true, y_pred):
        #    max_length_f = 1.0 * max_length
        #    y_pred = K.round(y_pred * max_length_f) / max_length_f
        #    y_pred.sort(axis=1)
        #    y_true.sort(axis=1)
        #    y_true = K.flatten(y_true)
        #    y_pred = K.flatten(y_pred)            
        #    return max_length_f * objectives.binary_crossentropy(y_true, y_pred)
        
        def t_loss(x_true_t, x_pred_t):
            max_length_f = 1.0 * max_length

            x_true_type = x_true_t[:,:,:conn_dim_start]
            x_pred_type = x_pred_t[:,:,:conn_dim_start]

            x_true_type = K.flatten(x_true_type)
            x_pred_type = K.flatten(x_pred_type)

            return max_length_f * objectives.binary_crossentropy(x_true_type, x_pred_type)

        def c_loss(x_true_c, x_pred_c):
            max_length_f = 1.0 * max_length
            x_true_conn = x_true_c[:,:,conn_dim_start:]
            x_pred_conn = x_pred_c[:,:,conn_dim_start:]

            #variant 1
            #x_true_conn = 0.5 * x_true_conn + 0.5
            #x_pred_conn = 0.5 * K.round(x_pred_conn * max_length_f) / max_length_f + 0.5
            #variant 2
            #x_pred_conn = 0.5 * x_pred_conn + 0.5
            #variant 3
            x_true_conn = K.round(x_true_conn * max_length_f)
            x_pred_conn = K.round(x_pred_conn * max_length_f)

            #x_pred_conn.sort(axis=2)
            #x_true_conn.sort(axis=2)

            #x_pred_conn = K.round(x_pred_conn * max_length_f) / max_length_f 
            x_true_conn = K.flatten(x_true_conn)
            x_pred_conn = K.flatten(x_pred_conn)

            return objectives.mean_squared_error(x_true_conn, x_pred_conn) / max_length_f

        def KL_loss(x_true_kl, x_pred_kl):
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return kl_loss

        #def vae_loss(x_true_vae, x_pred_vae):    
        #    #return  t_loss(x_true_vae, x_pred_vae) + KL_loss(x_true_vae, x_pred_vae)
        #    return 0.9 * t_loss(x_true_vae, x_pred_vae) + 0.1 * c_loss(x_true_vae, x_pred_vae) + KL_loss(x_true_vae, x_pred_vae)
        
        def vae_loss(x_true, x_pred):
            max_length_f = 1.0 * max_length

            x_true_type = x_true[:,:,:conn_dim_start]
            x_pred_type = x_pred[:,:,:conn_dim_start]
            x_true_conn = x_true[:,:,conn_dim_start:]
            x_pred_conn = x_pred[:,:,conn_dim_start:]

            #variant 1
            #x_true_conn = 0.5 * x_true_conn + 0.5
            #x_pred_conn = 0.5 * K.round(x_pred_conn * max_length_f) / max_length_f + 0.5
            #variant 2
            #x_pred_conn = K.round(x_pred_conn * max_length_f) / max_length_f 
            #variant 3
            x_true_conn = K.round(x_true_conn * max_length_f)
            x_pred_conn = K.round(x_pred_conn * max_length_f)

            y_true = K.concatenate((x_true_type, x_true_conn), axis = 2)
            y_pred = K.concatenate((x_pred_type, x_pred_conn), axis = 2)
            y_true = K.flatten(x_true)
            y_pred = K.flatten(x_pred)
            
            x_entropy = max_length_f * objectives.binary_crossentropy(y_true, y_pred)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)

            return x_entropy + kl_loss
            
        def type_acc(x_true_acc, x_pred_acc):
            y_true_t = x_true_acc[:,:,:conn_dim_start]
            y_pred_t = x_pred_acc[:,:,:conn_dim_start]
            y_true_t = K.flatten(y_true_t)
            y_pred_t = K.flatten(y_pred_t)
            return   K.mean(K.cast(K.equal(y_true_t, K.round(y_pred_t)), K.floatx()), axis=-1)
        #    #return K.cast(K.equal(K.argmax(y_true_t, axis=-1), K.argmax(y_pred_t, axis=-1)), K.floatx())

        #def vae_loss(x, x_decoded):
        #    x = K.flatten(x)
        #    x_decoded = K.flatten(x_decoded)
        #    xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded) 
        #    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
        #    return xent_loss + kl_loss

        return type_acc, t_loss, c_loss, vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])

    def _buildDecoder(self, z, latent_rep_size, max_length, connectivity_dims, intput_width):
        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        h = GRU(501, return_sequences = True, name='gru_2')(h)
        h = GRU(501, return_sequences = True, name='gru_3')(h)

        #return TimeDistributed(Dense(intput_width, activation='softmax'), name='decoded_mean')(h)

        d_type = MyCropping1D(cropping=(0, 300), name='crop_type')(h)
        #d_type = Dense(intput_width - connectivity_dims, name='dense_type')(h)
        #d_type = Convolution1D(intput_width - connectivity_dims, 1, activation = 'softmax', name='conv_type')(h)
        #d_type = Dense(intput_width - connectivity_dims, name='dense_type_1', activation = 'relu')(d_type)
        d_type = TimeDistributed(Dense(intput_width - connectivity_dims, activation='softmax'), name='decoded_type')(d_type)

        d_conn = MyCropping1D(cropping=(200, 0), name='crop_conn')(h)
        #d_conn = Dense(connectivity_dims, name='dense_conn_1')(h)
        d_conn = TimeDistributed(Dense(connectivity_dims,  activation = 'relu'), name='dense_conn_2')(d_conn)
        d_conn = Flatten(name='flatten_conn')(d_conn)
        d_conn = BatchNormalization(name='normalize_conn')(d_conn)
        #d_conn = Dense(max_length * connectivity_dims, name='dense_conn_2', activation = 'softmax')(d_conn)
        d_conn = Reshape((max_length, connectivity_dims), name='reshape_conn')(d_conn)
        #d_conn = Lambda(lambda x: (x - 0.5) * 2.0, name='lambda_conn', output_shape = (max_length, connectivity_dims))(d_conn)

        #d_conn = Convolution1D(connectivity_dims, 1, activation = 'relu', name='conv_conn')(h)
        #d_conn = Dense(connectivity_dims, name='dense_conn_1', activation = 'relu')(d_conn)
        
        #max_length_f = 1.0 * max_length
        #d_conn = Lambda(lambda x: K.round(x * max_length_f) / max_length_f, name='rounded_conn', output_shape = (max_length, connectivity_dims))(d_conn)

        return merge([d_type, d_conn], mode = 'concat')
        
        #h = merge([d_type, d_conn], mode = 'concat')
        #return TimeDistributed(Dense(intput_width), name='decoded_vec')(h)             

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, connectivity_dims = 0, latent_rep_size = 292):
        self.create(charset, connectivity_dims = connectivity_dims, weights_file = weights_file, latent_rep_size = latent_rep_size)
