__author__ = "James Large, Withington"

from tensorflow import keras
import numpy as np

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork


class DeepSpectraNetwork(BaseDeepNetwork):
    """DeepSpectra.

    Implemented based on the paper by Xiaolei Zhang et. al

    Network originally defined in:

    @article{zhang2019deepspectra,
      title={DeepSpectra: An end-to-end deep learning approach for quantitative spectral analysis},
      author={Zhang, Xiaolei and Lin, Tao and Xu, Jinfan and Luo, Xuan and Ying, Yibin},
      journal={Analytica chimica acta},
      volume={1058},
      pages={48--57},
      year={2019},
      publisher={Elsevier}
    }
    """

    def __init__(self,
                 kernel_size_conv1=7,  # aka kernel size 1. modal value taken as default
                 kernel_size_conv3_2=3,  # aka kernel size 2. modal value taken as default
                 kernel_size_conv3_3=5,  # aka kernel size 3. modal value taken as default
                 stride_conv1=3,  # aka stride 1, modal value
                 stride_conv3=2,  # aka stride 1, modal value
                 nb_dense_nodes=32,  # aka hidden number. modal value
                 dropout_rate=0.3,  # original paper varies per dataset, mean value
                 reg_coeff=0.001,
                 random_seed=0):
        '''
        :param kernel_size_conv1: int, size of the kernels in the first conv layer. AKA kernel size 1
        :param kernel_size_conv3_2: int, size of the kernels in the second set of filters of the third conv layer. AKA kernel size 2
        :param kernel_size_conv3_3: int, size of the kernels in the third set of filters of the third conv layer. AKA kernel size 3
        :param stride_conv1: int, the stride length of the first conv layer. AKA stride 1
        :param stride_conv3: int, the stride length of the second and third set of conv filters in the third conv layer. AKA stride 2
        :param nb_dense_nodes: int, the number of nodes in the penultimate fully connected layer. AKA hidden number
        :param dropout_rate: float, 0 to 1, the dropout rate for the penultimate fully connected layer.
        :param random_seed: int, seed to any needed random actions
        '''

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

        self.kernel_size_conv1 = kernel_size_conv1
        self.kernel_size_conv3_2 = kernel_size_conv3_2
        self.kernel_size_conv3_3 = kernel_size_conv3_3
        self.stride_conv1 = stride_conv1
        self.stride_conv3 = stride_conv3
        self.nb_dense_nodes = nb_dense_nodes
        self.dropout_rate = dropout_rate
        self.reg_coeff = reg_coeff

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        padding = 'same'
        conv1_filters = 8
        conv2_filters = 4
        conv3_filters = 4

        input_layer = keras.layers.Input(input_shape)

        # Conv1
        conv1 = keras.layers.Conv1D(filters=conv1_filters,
                                    kernel_size=self.kernel_size_conv1,
                                    strides=self.stride_conv1,
                                    kernel_regularizer=keras.regularizers.l2(self.reg_coeff),
                                    padding=padding)(input_layer)

        # Conv2
        conv2_1 = keras.layers.Conv1D(filters=conv2_filters,
                                      kernel_size=1,
                                      kernel_regularizer=keras.regularizers.l2(self.reg_coeff),
                                      padding=padding,
                                      activation=keras.layers.LeakyReLU())(conv1)
        conv2_2 = keras.layers.Conv1D(filters=conv2_filters,
                                      kernel_size=1,
                                      kernel_regularizer=keras.regularizers.l2(self.reg_coeff),
                                      padding=padding,
                                      activation=keras.layers.LeakyReLU())(conv1)
        conv2_3 = keras.layers.MaxPool1D(pool_size=2)(conv1)

        # Conv3
        conv3_1 = keras.layers.Conv1D(filters=conv3_filters,
                                      kernel_size=1,
                                      padding=padding,
                                      strides=self.stride_conv3,
                                      kernel_regularizer=keras.regularizers.l2(self.reg_coeff),
                                      activation=keras.layers.LeakyReLU())(conv1)  # shortcut from conv1
        conv3_2 = keras.layers.Conv1D(filters=conv3_filters,
                                      kernel_size=self.kernel_size_conv3_2,
                                      strides=self.stride_conv3,
                                      kernel_regularizer=keras.regularizers.l2(self.reg_coeff),
                                      padding=padding,
                                      activation=keras.layers.LeakyReLU())(conv2_1)
        conv3_3 = keras.layers.Conv1D(filters=conv3_filters,
                                      kernel_size=self.kernel_size_conv3_3,
                                      strides=self.stride_conv3,
                                      kernel_regularizer=keras.regularizers.l2(self.reg_coeff),
                                      padding=padding,
                                      activation=keras.layers.LeakyReLU())(conv2_2)
        conv3_4 = keras.layers.Conv1D(filters=conv3_filters,
                                      kernel_size=1,
                                      kernel_regularizer=keras.regularizers.l2(self.reg_coeff),
                                      padding=padding,
                                      activation=keras.layers.LeakyReLU())(conv2_3)

        # Flatten
        flatten = keras.layers.Concatenate(axis=2)([conv3_1, conv3_2, conv3_3, conv3_4])
        flatten = keras.layers.Flatten()(flatten)
        flatten = keras.layers.BatchNormalization()(flatten)

        # F1
        dense = keras.layers.Dense(units=self.nb_dense_nodes,
                                   kernel_regularizer=keras.regularizers.l2(self.reg_coeff))(flatten)
        dense = keras.layers.BatchNormalization()(dense)
        dense = keras.layers.Activation(keras.layers.LeakyReLU())(dense)
        dense = keras.layers.Dropout(rate=self.dropout_rate)(dense)

        # 'F2' or 'output' added by estimator
        return input_layer, dense
