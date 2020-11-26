import numpy as np
import numpy.random as rng
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from own_package.data_augmentation import produce_smote, produce_invariant


class Shortcut_fl:
    def __init__(self, features_c, labels, scaler, feature_names, label_names, norm_mask):
        self.features_c_names = feature_names
        self.features_c = features_c
        self.features_c_dim = features_c.shape[1]
        self.count = features_c.shape[0]
        self.norm_mask = norm_mask
        mask = np.array([1] * self.features_c_dim, dtype=np.bool)
        mask[norm_mask] = 0
        if features_c[:, mask].shape[1] == 0:
            self.scaler = 0
            self.features_c_norm = features_c
        else:
            if scaler is None:
                # If scaler is None, means normalize the data with all input data
                self.scaler = MinMaxScaler()
                self.scaler.fit(features_c[:, mask])  # Setting up scaler
            else:
                # If scaler is given, means normalize the data with the given scaler
                self.scaler = scaler
            features_c_norm = self.scaler.transform(features_c[:, mask])  # Normalizing features_c
            self.features_c_norm = np.copy(features_c)
            self.features_c_norm[:, mask] = features_c_norm
        self.labels = labels
        self.labels_names = label_names


def load_testset_to_fl(testset_excel_file, norm_mask, scaler):
    df = pd.read_excel(testset_excel_file, index_col=0)
    features = df.iloc[:,:6].values
    labels = df.iloc[:,6:].values
    return Shortcut_fl(features_c=features, labels=labels, scaler=scaler,
                       feature_names=df.columns[:6], label_names=df.columns[6:], norm_mask=norm_mask)


def load_data_to_fl(data_loader_excel_file, normalise_labels, norm_mask=None):
    # Read in the features and labels worksheet into dataframe
    xls = pd.ExcelFile(data_loader_excel_file)
    df_features = pd.read_excel(xls, sheet_name='features', index_col=0)
    df_labels = pd.read_excel(xls, sheet_name='cutoff', index_col=0)

    features_c = df_features.values
    features_c_names = df_features.columns.values

    labels = df_labels.values
    labels_names = df_labels.columns.values

    fl = Features_labels(features_c, labels,
                         features_c_names=features_c_names, labels_names=labels_names,
                         norm_mask=norm_mask, normalise_labels=normalise_labels,)

    return fl


class Features_labels:
    def __init__(self, features_c, labels, features_c_names=None, labels_names=None, scaler=None,
                 norm_mask=None, normalise_labels=False, labels_scaler=None, idx=None):
        """
        Creates fl class with a lot useful attributes
        :param features_c: Continuous features. Np array, no. of examples x continous features
        :param labels: Labels as np array, no. of examples x dim
        :param scaler: Scaler to transform features c. If given, use given MinMax scaler from sklearn,
        else create scaler based on given features c.
        """
        # Setting up features
        self.count = features_c.shape[0]
        # idx is for when doing k-fold cross validation, to keep track of which examples are in the val. set
        if isinstance(idx, np.ndarray):
            self.idx = idx
        else:
            self.idx = np.arange(self.count)
        self.features_c = np.copy(features_c)
        self.features_c_dim = features_c.shape[1]
        self.features_c_names = features_c_names

        if norm_mask:
            '''
            The norm mask is a list which indicates which features are normalize or not
            For example, our 6 features are the 2 compositions, 1 thickness, and the dimension (which is 3 features
            since we use 1 hot encoding for 0D, 1D, 2D)
            We want to normalize the compositions and thickness but not the dimensions so we use the following 
            norm mask = [0, 1, 3, 4, 5] which prevents all but the 2nd entry feature (thickness) from being normalized
            '''
            self.norm_mask = norm_mask
            mask = np.array([1] * self.features_c_dim, dtype=np.bool)
            mask[norm_mask] = 0  # Entries with 1 are set to 0 so that they are not normalized
            if features_c[:, mask].shape[1] == 0:  # Means that all features are masked and not normalized
                self.scaler = 0
                self.features_c_norm = features_c
            else:
                if scaler is None:
                    # If scaler is None, means normalize the data with all input data
                    self.scaler = MinMaxScaler()
                    self.scaler.fit(features_c[:, mask])  # Setting up scaler
                else:
                    # If scaler is given, means normalize the data with the given scaler
                    # This is used when the fl class is being made for k-fold cross validation where we want each fold
                    # to have the same normalization so we pass the scaler from the original fl class to each fold's fl
                    self.scaler = scaler
                features_c_norm = self.scaler.transform(features_c[:, mask])  # Normalizing features_c
                self.features_c_norm = features_c
                self.features_c_norm[:, mask] = features_c_norm
        else:
            # Normalizing all features
            self.norm_mask = None
            if scaler is None:
                # If scaler is None, means normalize the data with all input data
                self.scaler = MinMaxScaler()
                self.scaler.fit(features_c)  # Setting up scaler
            else:
                # If scaler is given, means normalize the data with the given scaler
                self.scaler = scaler
            self.features_c_norm = self.scaler.transform(features_c)  # Normalizing features_c

        # Setting up labels
        self.labels = labels
        if len(labels.shape) == 2:
            self.labels_dim = labels.shape[1]
        else:
            self.labels_dim = 1
        self.labels_names = labels_names

        if normalise_labels:
            self.normalise_labels = normalise_labels
            if labels_scaler is None:
                self.labels_scaler = MinMaxScaler(feature_range=(0, 1))
                self.labels_scaler.fit(labels)
            else:
                self.labels_scaler = labels_scaler
            self.labels_norm = self.labels_scaler.transform(labels)
        else:
            self.normalise_labels = False
            self.labels_scaler = None
            self.labels_norm = None

    def apply_scaling(self, features_c):
        if features_c.ndim == 1:
            features_c = features_c.reshape((1, -1))
        if self.norm_mask:
            norm_mask = self.norm_mask
            mask = np.array([1] * self.features_c_dim, dtype=np.bool)
            mask[norm_mask] = 0
            features_c_norm = np.copy(features_c)
            features_c_norm[:, mask] = self.scaler.transform(features_c[:, mask])
        else:
            features_c_norm = self.scaler.transform(features_c)
        return features_c_norm

    def generate_random_examples(self, numel):
        gen_features_c_norm_a = rng.random_sample((numel, self.features_c_dim))
        gen_features_c_a = self.scaler.inverse_transform(gen_features_c_norm_a)

        # Creating dic for SNN prediction
        gen_dic = {}
        gen_dic = dict(
            zip(('gen_features_c_a', 'gen_features_c_norm_a'), (gen_features_c_a, gen_features_c_norm_a)))
        return gen_dic

    def create_kf(self, k_folds, shuffle=True):
        '''
        Almost the same as skf except can work for regression labels and folds are not stratified.
        Create list of tuples containing (fl_train,fl_val) fl objects for k fold cross validation
        :param k_folds: Number of folds
        :return: List of tuples
        '''
        fl_store = []
        # Instantiate the cross validator
        skf = KFold(n_splits=k_folds, shuffle=shuffle)
        # Loop through the indices the split() method returns
        for _, (train_indices, val_indices) in enumerate(skf.split(self.features_c, self.labels)):
            # Generate batches from indices
            xval_idx = self.idx[val_indices]
            xtrain, xval = self.features_c[train_indices], self.features_c[val_indices]
            ytrain, yval = self.labels[train_indices], self.labels[val_indices]
            fl_store.append(
                (Features_labels(xtrain, ytrain, scaler=self.scaler, normalise_labels=self.normalise_labels,
                                 labels_scaler=self.labels_scaler,labels_names=self.labels_names,
                                 norm_mask=self.norm_mask, features_c_names=self.features_c_names),
                 Features_labels(xval, yval, idx=xval_idx, scaler=self.scaler, normalise_labels=self.normalise_labels,
                                 labels_scaler=self.labels_scaler, labels_names=self.labels_names,
                                 norm_mask=self.norm_mask, features_c_names=self.features_c_names))
            )
        return fl_store

    def smote_kf_augment(self, smote_excel, k_folds, shuffle=True):
        """
        Same as kf above. But appends all smote data to each fold's training examples. Validation examples no change.
        :param smote_excel:
        :param k_folds:
        :param shuffle:
        :return:
        """
        smote = pd.read_excel(smote_excel, index_col=0).values
        smote_features = smote[:, :6]
        smote_labels = smote[:,6:]
        fl_store = []
        # Instantiate the cross validator
        skf = KFold(n_splits=k_folds, shuffle=shuffle)
        # Loop through the indices the split() method returns
        for _, (train_indices, val_indices) in enumerate(skf.split(self.features_c, self.labels)):
            # Generate batches from indices
            xval_idx = self.idx[val_indices]
            xtrain, xval = np.concatenate((self.features_c[train_indices], smote_features), axis=0), self.features_c[val_indices]
            ytrain, yval = np.concatenate((self.labels[train_indices], smote_labels), axis=0), self.labels[val_indices]
            fl_store.append(
                (Features_labels(xtrain, ytrain, scaler=self.scaler, normalise_labels=self.normalise_labels,
                                 labels_scaler=self.labels_scaler,labels_names=self.labels_names,
                                 norm_mask=self.norm_mask, features_c_names=self.features_c_names),
                 Features_labels(xval, yval, idx=xval_idx, scaler=self.scaler, normalise_labels=self.normalise_labels,
                                 labels_scaler=self.labels_scaler, labels_names=self.labels_names,
                                 norm_mask=self.norm_mask, features_c_names=self.features_c_names))
            )
        return fl_store

    def fold_smote_kf_augment(self, numel, k_folds, shuffle=True):
        """
        Same as kf above. But appends all smote data to each fold's training examples. Validation examples no change.
        :param smote_excel:
        :param k_folds:
        :param shuffle:
        :return:
        """
        fl_store = []
        # Instantiate the cross validator
        skf = KFold(n_splits=k_folds, shuffle=shuffle)
        # Loop through the indices the split() method returns
        for _, (train_indices, val_indices) in enumerate(skf.split(self.features_c, self.labels)):
            # Generate batches from indices
            xval_idx = self.idx[val_indices]
            smote_features, smote_labels = produce_smote(self.features_c[train_indices],
                                                         self.labels[train_indices], numel=numel)
            xtrain, xval = np.concatenate((self.features_c[train_indices], smote_features), axis=0), self.features_c[val_indices]
            ytrain, yval = np.concatenate((self.labels[train_indices], smote_labels), axis=0), self.labels[val_indices]
            fl_store.append(
                (Features_labels(xtrain, ytrain, scaler=self.scaler, normalise_labels=self.normalise_labels,
                                 labels_scaler=self.labels_scaler,labels_names=self.labels_names,
                                 norm_mask=self.norm_mask, features_c_names=self.features_c_names),
                 Features_labels(xval, yval, idx=xval_idx, scaler=self.scaler, normalise_labels=self.normalise_labels,
                                 labels_scaler=self.labels_scaler, labels_names=self.labels_names,
                                 norm_mask=self.norm_mask, features_c_names=self.features_c_names))
            )
        return fl_store

    def fold_invariant_kf_augment(self, numel, k_folds, shuffle=True):
        """
        Same as kf above. But appends all smote data to each fold's training examples. Validation examples no change.
        :param smote_excel:
        :param k_folds:
        :param shuffle:
        :return:
        """
        fl_store = []
        # Instantiate the cross validator
        skf = KFold(n_splits=k_folds, shuffle=shuffle)
        # Loop through the indices the split() method returns
        for _, (train_indices, val_indices) in enumerate(skf.split(self.features_c, self.labels)):
            # Generate batches from indices
            xval_idx = self.idx[val_indices]
            smote_features, smote_labels = produce_invariant(self.features_c[train_indices],
                                                         self.labels[train_indices], numel=numel)
            xtrain, xval = np.concatenate((self.features_c[train_indices], smote_features), axis=0), self.features_c[val_indices]
            ytrain, yval = np.concatenate((self.labels[train_indices], smote_labels), axis=0), self.labels[val_indices]
            fl_store.append(
                (Features_labels(xtrain, ytrain, scaler=self.scaler, normalise_labels=self.normalise_labels,
                                 labels_scaler=self.labels_scaler,labels_names=self.labels_names,
                                 norm_mask=self.norm_mask, features_c_names=self.features_c_names),
                 Features_labels(xval, yval, idx=xval_idx, scaler=self.scaler, normalise_labels=self.normalise_labels,
                                 labels_scaler=self.labels_scaler, labels_names=self.labels_names,
                                 norm_mask=self.norm_mask, features_c_names=self.features_c_names))
            )
        return fl_store


class Features_labels_grid:
    def __init__(self, features, labels, idx=None):
        """
        Creates fl class with a lot useful attributes for grid data classification
        :param features:
        :param labels: Labels as np array, no. of examples x dim
        :param idx: Used to keep track of the example index when performing k-fold cv
        """
        # Setting up features
        self.count = features.shape[0]
        self.features = np.copy(features)
        self.features_dim = features.shape[1]

        # idx is for when doing k-fold cross validation, to keep track of which examples are in the val. set
        if isinstance(idx, np.ndarray):
            self.idx = idx
        else:
            self.idx = np.arange(self.count)  # If idx is none, means this is the original fl and make new index list

        # Setting up labels
        self.labels = np.array(labels)
        self.labels_dim = 1

    def create_kf(self, k_folds, shuffle=True):
        '''
        Create list of tuples containing (fl_train,fl_val) fl objects for k fold cross validation
        fl_train contains the training example and fl_val contains the validation example for that current folf
        :param k_folds: Number of folds
        :param shuffle: Whether to shuffle the data before splitting into the k-folds. Usually set to True.
        :return: List of tuples of (fl_train,fl_val)
        '''
        fl_store = []
        # Instantiate the cross validator
        skf = KFold(n_splits=k_folds, shuffle=shuffle)
        # Loop through the indices the split() method returns
        for _, (train_indices, val_indices) in enumerate(skf.split(self.features, self.labels)):
            # Generate batches from indices
            xval_idx = self.idx[val_indices]  # Keep track of which idx the validation data is
            xtrain, xval = self.features[train_indices], self.features[val_indices]
            ytrain, yval = self.labels[train_indices], self.labels[val_indices]
            fl_store.append((Features_labels_grid(xtrain, ytrain),
                             Features_labels_grid(xval, yval, idx=xval_idx)))
        return fl_store
