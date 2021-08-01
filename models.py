from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import MeanAbsolutePercentageError
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression



def create_hparams(learning_rate=0.001, optimizer='Adam', epochs=100, batch_size=64,
                   activation='relu', loss='mre', nodes=10, reg_l1=0, reg_l2=0,gamma=1, C=0.1, chain=False,
                   max_depth=6, num_est=300, verbose=1):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['learning_rate', 'optimizer', 'epochs', 'batch_size',
             'activation', 'loss', 'nodes', 'reg_l1', 'reg_l2', 'gamma', 'C', 'chain',
             'max_depth', 'num_est', 'verbose']
    values = [learning_rate, optimizer, epochs, batch_size,
              activation, loss, nodes, reg_l1, reg_l2, gamma, C, chain,
              max_depth, num_est, verbose]
    hparams = dict(zip(names, values))
    return hparams


class Kmodel:
    def __init__(self, fl, hparams):
        """
        Initialises new ANN model
        :param fl: fl class
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function
        """
        self.features_dim = fl.features_c_dim
        self.labels_dim = fl.labels_dim  # Assuming that each task has only 1 dimensional output
        self.hparams = hparams
        self.normalise_labels = fl.normalise_labels
        self.labels_scaler = fl.labels_scaler
        features_in = Input(shape=(self.features_dim,), name='main_features_c_input')
        # Build keras ANN model
        x = Dense(units=hparams['nodes'],
                  activation=hparams['activation'],
                  kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                  name='Layer_' + str(0))(features_in)
        x = Dense(units=hparams['nodes'],
                  activation=hparams['activation'],
                  kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                  name='Layer_' + str(1))(x)
        x = Dense(units=hparams['nodes'],
                  activation=hparams['activation'],
                  kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                  name='Layer_' + str(2))(x)
        # x = BatchNormalization()(x)
        x = Dense(units=self.labels_dim,
                  activation='linear',
                  kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                  name='Final')(x)
        self.model = Model(inputs=features_in, outputs=x)
        optimizer = Adam(learning_rate=hparams['learning_rate'], clipnorm=1)

        def mean_relative_error(y_true, y_pred):
            diff = K.abs((y_true - y_pred) / K.reshape(K.clip(K.abs(y_true[:,-1]),
                                                    K.epsilon(),
                                                    None), (-1,1)))
            return 100. * K.mean(diff, axis=-1)

        if hparams['loss'] == 'mape':
            self.model.compile(optimizer=optimizer, loss=MeanAbsolutePercentageError())
        elif hparams['loss'] == 'mre':
            self.model.compile(optimizer=optimizer, loss=mean_relative_error)
        elif hparams['loss'] == 'mse':
            self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        # self.model.summary()

    def train_model(self, fl, i_fl,
                    save_name='mt.h5', save_dir='./save/models/',
                    save_mode=False, plot_name=None):
        # Training model
        training_features = fl.features_c_norm
        val_features = i_fl.features_c_norm
        if self.normalise_labels:
            training_labels = fl.labels_norm
            val_labels = i_fl.labels_norm
        else:
            training_labels = fl.labels
            val_labels = i_fl.labels

        # Plotting
        if plot_name:
            history = self.model.fit(training_features, training_labels,
                                     validation_data=(val_features, val_labels),
                                     epochs=self.hparams['epochs'],
                                     batch_size=self.hparams['batch_size'],
                                     verbose=self.hparams['verbose'])
            # Debugging check to see features and prediction
            # pprint.pprint(training_features)
            # pprint.pprint(self.model.predict(training_features))
            # pprint.pprint(training_labels)

            # summarize history for accuracy
            plt.semilogy(history.history['loss'], label=['train'])
            plt.semilogy(history.history['val_loss'], label=['test'])
            plt.plot([],[],' ',label='Final train: {:.3e}'.format(history.history['loss'][-1]))
            plt.plot([], [], ' ', label='Final val: {:.3e}'.format(history.history['val_loss'][-1]))
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(loc='upper right')
            plt.savefig(plot_name, bbox_inches='tight')
            plt.close()
        else:
            history = self.model.fit(training_features, training_labels,
                                     epochs=self.hparams['epochs'],
                                     batch_size=self.hparams['batch_size'],
                                     verbose=self.hparams['verbose'])
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        return self

    def predict(self, eval_fl):
        features = eval_fl.features_c_norm
        predictions = self.model.predict(features)
        return predictions  # If labels is normalized, the prediction here is also normalized!


class DTRmodel:
    def __init__(self, fl, max_depth=8, num_est=300, chain=False):
        """
        Initialises new DTR model
        :param fl: fl class
        :param max_depth: max depth of each tree
        :param num_est: Number of estimators in the ensemble of trees
        :param chain: regressor chain (True) or independent multi-output (False)
        """
        self.labels_dim = fl.labels_dim
        self.labels_scaler = fl.labels_scaler
        if chain:
            self.model = RegressorChain(
                AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth), n_estimators=num_est))
        else:
            self.model = MultiOutputRegressor(
                AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth), n_estimators=num_est))
        self.normalise_labels = fl.normalise_labels

    def train_model(self, fl, *args, **kwargs):
        # *args, **kwargs is there for compatibility with the KModel class
        training_features = fl.features_c_norm
        if self.normalise_labels:
            training_labels = fl.labels_norm
        else:
            training_labels = fl.labels
        self.model.fit(training_features, training_labels)
        return self

    def predict(self, eval_fl):
        features = eval_fl.features_c_norm
        if self.labels_dim == 1:  # If labels is 1D output, the prediction will be a 1D array not 2D
            y_pred = self.model.predict(features)[:, None]
        else:
            y_pred = self.model.predict(features)
        return y_pred  # If labels is normalized, the prediction here is also normalized!


class SVRmodel:
    def __init__(self, fl, gamma, C):
        """
        Initialises new DTR model
        :param fl: fl class
        :param gamma
        :param C
        """
        self.labels_dim = fl.labels_dim
        self.labels_scaler = fl.labels_scaler
        self.model = MultiOutputRegressor(SVR(gamma=gamma, C=C, kernel='rbf'))
        self.normalise_labels = fl.normalise_labels

    def train_model(self, fl, *args, **kwargs):
        # *args, **kwargs is there for compatibility with the KModel class
        training_features = fl.features_c_norm
        if self.normalise_labels:
            training_labels = fl.labels_norm
        else:
            training_labels = fl.labels
        self.model.fit(training_features, training_labels)
        return self

    def predict(self, eval_fl):
        features = eval_fl.features_c_norm
        if self.labels_dim == 1:  # If labels is 1D output, the prediction will be a 1D array not 2D
            y_pred = self.model.predict(features)[:, None]
        else:
            y_pred = self.model.predict(features)
        return y_pred  # If labels is normalized, the prediction here is also normalized!


class LRmodel:
    def __init__(self, fl):
        """
        Initialises new DTR model
        :param fl: fl class
        :param gamma
        :param C
        """
        self.labels_dim = fl.labels_dim
        self.labels_scaler = fl.labels_scaler
        self.model = LinearRegression()
        self.normalise_labels = fl.normalise_labels

    def train_model(self, fl, *args, **kwargs):
        # *args, **kwargs is there for compatibility with the KModel class
        training_features = fl.features_c_norm
        if self.normalise_labels:
            training_labels = fl.labels_norm
        else:
            training_labels = fl.labels
        self.model.fit(training_features, training_labels)
        return self

    def predict(self, eval_fl):
        features = eval_fl.features_c_norm
        if self.labels_dim == 1:  # If labels is 1D output, the prediction will be a 1D array not 2D
            y_pred = self.model.predict(features)[:, None]
        else:
            y_pred = self.model.predict(features)
        return y_pred  # If labels is normalized, the prediction here is also normalized!

