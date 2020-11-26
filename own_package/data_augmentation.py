import numpy as np
import itertools, random
import six
import sys
sys.modules['sklearn.externals.six'] = six
from imblearn.over_sampling import SMOTE


def produce_smote(features, labels, numel):
    '''
    Features should contain only composition and thickness. SMOTE for each dimension separately
    '''
    data_store = []
    for colidx in [-3, -2, -1]:
        dim_idx = np.where(features[:, colidx] == 1)[0]
        data_store.append(np.concatenate((features[dim_idx, :-3], labels[dim_idx, :]), axis=1))

    data_smote_all = []
    for dim, data2Del in enumerate(data_store):
        ind_list = [i for i in range(data2Del.shape[0])]
        ind_set = list(itertools.combinations(ind_list, 3))
        num_original = len(ind_list)
        iter_required = int(numel / (num_original - 3))
        num_comb = len(ind_set)
        jump = int(num_comb / iter_required)
        model_smote = SMOTE(k_neighbors=2, random_state=0)
        data_smote_all_single_dim = []

        for i in range(0, num_comb, jump):
            item = ind_set[i]
            ind_ = list(item)
            y_smote = np.zeros(data2Del.shape[0])
            y_smote[ind_] = 1
            data_smote_resampled, y_smote_resampled = model_smote.fit_resample(np.array(data2Del), y_smote)
            ind = np.where(y_smote_resampled == 1)
            data_ = data_smote_resampled[ind].tolist()
            data_smote_all_single_dim.extend(data_)

        dim_features = [0, 0, 0]
        dim_features[dim] = 1
        data_smote_all_single_dim = [data[:-3] + dim_features + data[-3:] for data in data_smote_all_single_dim]
        data_smote_all.extend(data_smote_all_single_dim)

    data_smote_all = np.unique(np.array(data_smote_all), axis=0)
    # Split features and labels
    return data_smote_all[:, :features.shape[1]], data_smote_all[:, features.shape[1]:]


def produce_invariant(features, labels, numel):
    feature_store = []
    label_store = []
    allowed_variation = [0.02, 0.02, 5]  # variation for CNT, PVA, thickness
    for feature, label in zip(features.tolist(), labels.tolist()):
        for _ in range(numel):
            new_feature = feature[:]
            rand = [random.uniform(-1, 1) for _ in range(3)]
            for idx, (x, r, c) in enumerate(zip(feature, rand, allowed_variation)):
                new_x = x + r * c
                new_feature[idx] = max(new_x, 0)
                pass
            feature_store.append(new_feature)
            label_store.append(label)
    return np.array(feature_store), np.array(label_store)
