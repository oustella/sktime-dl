__author__ = "James Large"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepRegressor
from sktime_dl.deeplearning.deepspectra._base import DeepSpectraNetwork
from sktime_dl.utils import check_and_clean_data


class DeepSpectraRegressor(BaseDeepRegressor, DeepSpectraNetwork):
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

    # todo add regularisation to obj func
    """

    def __init__(self,
                 nb_epochs=2000,
                 batch_size=32,
                 kernel_size_conv1=7,  # aka kernel size 1. modal value taken as default
                 kernel_size_conv3_2=3,  # aka kernel size 2. modal value taken as default
                 kernel_size_conv3_3=5,  # aka kernel size 3. modal value taken as default
                 stride_conv1=3,  # aka stride 1, modal value
                 stride_conv3=2,  # aka stride 1, modal value
                 nb_dense_nodes=32,  # aka hidden number. modal value
                 dropout_rate=0.3,  # original paper varies per dataset, mean value
                 reg_coeff=0.001,  # varies between 0.001 and 0.01
                 learning_rate=0.01,
                 learning_rate_decay=0.001,

                 callbacks=None,
                 random_seed=0,
                 verbose=False,
                 model_name="cnn_regressor",
                 model_save_directory=None):
        super().__init__(
            model_name=model_name,
            model_save_directory=model_save_directory)
        DeepSpectraNetwork.__init__(
            self,
            kernel_size_conv1=kernel_size_conv1,  # aka kernel size 1. modal value taken as default
            kernel_size_conv3_2=kernel_size_conv3_2,  # aka kernel size 2. modal value taken as default
            kernel_size_conv3_3=kernel_size_conv3_3,  # aka kernel size 3. modal value taken as default
            stride_conv1=stride_conv1,  # aka stride 1, modal value
            stride_conv3=stride_conv3,  # aka stride 1, modal value
            nb_dense_nodes=nb_dense_nodes,  # aka hidden number. modal value
            dropout_rate=dropout_rate,  # original paper varies per dataset, mean value
            random_seed=random_seed)
        '''
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param kernel_size_conv1: int, size of the kernels in the first conv layer. AKA kernel size 1 
        :param kernel_size_conv3_2: int, size of the kernels in the second set of filters of the third conv layer. AKA kernel size 2
        :param kernel_size_conv3_3: int, size of the kernels in the third set of filters of the third conv layer. AKA kernel size 3
        :param stride_conv1: int, the stride length of the first conv layer. AKA stride 1
        :param stride_conv3: int, the stride length of the second and third set of conv filters in the third conv layer. AKA stride 2
        :param nb_dense_nodes: int, the number of nodes in the penultimate fully connected layer. AKA hidden number
        :param dropout_rate: float, 0 to 1, the dropout rate for the penultimate fully connected layer.  
        
        :param reg_coeff: float, 0 to 1, the dropout rate for the penultimate fully connected layer.  
        :param learning_rate: float, the learning rate controlling the size of update steps
        :param learning_rate_decay: float, the decay applied to the learning rate over time. Authors may have had a similar situation to this https://github.com/XifengGuo/CapsNet-Keras/issues/9
        
        :param callbacks: list of tf.keras.callbacks.Callback objects
        :param random_seed: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and file writing purposes
        :param model_save_directory: string, if not None; location to save the trained keras model in hdf5 format
        '''
        self.verbose = verbose
        self.is_fitted = False

        self.callbacks = callbacks if callbacks is not None else []

        self.input_shape = None
        self.history = None

        self.nb_epochs = nb_epochs
        self.batch_size = batch_size

        self.reg_coeff = reg_coeff
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        output : a compiled Keras Model
        """
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(units=1)(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, decay=self.learning_rate_decay),
                      metrics=['mean_squared_error'])

        return model

    def fit(self, X, y, input_checks=True, **kwargs):
        """
        Build the regressor on the training set (X, y)
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame of Series objects is passed, column 0 is extracted.
        y : array-like, shape = [n_instances]
            The regression values.
        input_checks: boolean
            whether to check the X and y parameters
        Returns
        -------
        self : object
        """
        X = check_and_clean_data(X, y, input_checks=input_checks)

        # ignore the number of instances, X.shape[0], just want the shape of each instance
        self.input_shape = X.shape[1:]

        self.model = self.build_model(self.input_shape)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y, batch_size=self.batch_size, epochs=self.nb_epochs,
                                      verbose=self.verbose, callbacks=self.callbacks)

        self.save_trained_model()
        self.is_fitted = True

        return self


if __name__ == "__main__":
    from sktime_dl.deeplearning.tests.test_regressors import test_regressor, test_regressor_forecasting

    test_regressor(DeepSpectraRegressor(nb_epochs=5))
    test_regressor_forecasting(DeepSpectraRegressor(nb_epochs=5))
