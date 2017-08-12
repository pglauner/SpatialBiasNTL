import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import randint
import pickle

from get_data import get_raw_ID_UC, get_raw_inspections
from tree_learning import run_randomsearch
from config import FEATURES, COMPOUND_FEATURES, LEVELS

param_dist = {"criterion": ["gini", "entropy"],
              "min_samples_split": randint(2, 20),
              "max_depth": randint(1, 20),
              "min_samples_leaf": randint(1, 20),
              "max_leaf_nodes": randint(2, 20)}


def get_level_cols(level=1):
    cols = []
    for i in xrange(1, level + 1):
        cols.append(LEVELS[i])

    return cols


# Change this global parameter
LEVEL = 1


def get_labeled_data(cols):
    level_cols = get_level_cols(LEVEL)
    all_UCs = get_raw_ID_UC(cols + level_cols)

    # Make sure to be columns in the same order as in cols
    all_UCs = all_UCs[['ID_UC'] + cols + level_cols]
    print '# all ID_UCs: {0}'.format(len(all_UCs))

    all_UCs = all_UCs[~all_UCs.isin(['nulo']).any(axis=1)]
    print '# ID_UCs without nulo: {0}'.format(len(all_UCs))

    # Convert level columns to int, some may be strings or nulo (filtered before)
    for level_col in ['ID_UC'] + level_cols:
        all_UCs[level_col] = all_UCs[level_col].astype(int)

    # Keep only one inspection per valid ID_UC
    inspected_UCs = get_raw_inspections().drop_duplicates('ID_UC')
    inspected_UCs = inspected_UCs[inspected_UCs.ID_UC.isin(all_UCs.ID_UC)]
    print '# inspected ID_UCs: {0}'.format(len(inspected_UCs))

    uninspected_UCs = DataFrame(all_UCs[~all_UCs.ID_UC.isin(inspected_UCs.ID_UC)].ID_UC)
    print '# uninspected ID_UCs: {0}'.format(len(uninspected_UCs))

    # Add labels
    all_UCs['s'] = np.where(all_UCs.ID_UC.isin(inspected_UCs.ID_UC), 1, 0)

    # Give consecutive index to avoid strange problem that appear before...
    return all_UCs.reset_index()


def get_splits(X):
    level_cols = get_level_cols(LEVEL)

    return X.groupby(level_cols).groups


def get_converted_feature(name, X):
    print 'Converting categorial to one-hot coding: {0}'.format(name)
    if 'force-one-hot' in FEATURES[name] and FEATURES[name]['force-one-hot']:
        enc = OneHotEncoder()
        X = enc.fit_transform(X).todense()
    else:
        dv = DictVectorizer(sparse=False)
        df = pd.DataFrame(X).convert_objects()
        X = dv.fit_transform(df.to_dict(orient='records'))

    return X


def execute_learning(name, X, y):
    print 'Shape of X: {0}'.format(X.shape)
    print 'Shape of y: {0}'.format(y.shape)

    try:
        clf = tree.DecisionTreeClassifier(class_weight='balanced')
        return run_randomsearch(X, y, clf, param_dist, cv=10, n_iter_search=100)
    except:
        print 'Exception'
        return 'Exception'


def main_single_feature():
    # TODO: all fields
    for name in ['xy']:
    #for name in FEATURES.keys():
        print 'Run local experiments for feature {0}, level {1}'.format(name, LEVEL)

        columns = FEATURES[name]['cols']

        # Get features and label for training
        data = get_labeled_data(columns)
        values = data.values[:, 1:]
        n = len(columns)
        X = values[:, :n]
        y = values[:, -1].astype(int)

        X = get_converted_feature(name, X)

        splits = get_splits(data)
        res = {}
        for i, split_key in enumerate(splits):
            print
            print 'Run experiments for feature {0} for key {1} ({2}/{3} keys)'.format(name, split_key, i + 1,
                                                                                      len(splits))
            selected_idxs = splits[split_key]
            selected_UCs = set(data[data.index.isin(selected_idxs)].ID_UC)
            Xs = X[selected_idxs, :]
            ys = y[selected_idxs]

            train_res = execute_learning(name, Xs, ys)
            res[split_key] = {'ID_UCs': selected_UCs,
                              'res': train_res}

        pickle.dump(res, open('bias_local_{0}_level_{1}.p'.format(name, LEVEL), 'wb'))


if __name__ == '__main__':
    main_single_feature()
