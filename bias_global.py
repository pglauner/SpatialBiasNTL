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
from config import FEATURES, COMPOUND_FEATURES

param_dist = {"criterion": ["gini", "entropy"],
              "min_samples_split": randint(2, 20),
              "max_depth": randint(1, 20),
              "min_samples_leaf": randint(1, 20),
              "max_leaf_nodes": randint(2, 20)}


def get_labeled_data(cols):
    all_UCs = get_raw_ID_UC(cols)
    # Make sure to be columns in the same order as in cols
    all_UCs = all_UCs[['ID_UC'] + cols]
    print '# all ID_UCs: {0}'.format(len(all_UCs))

    all_UCs = all_UCs[~all_UCs.isin(['nulo']).any(axis=1)]
    print '# ID_UCs without nulo: {0}'.format(len(all_UCs))

    # Keep only one inspection per valid ID_UC
    inspected_UCs = get_raw_inspections().drop_duplicates('ID_UC')
    inspected_UCs = inspected_UCs[inspected_UCs.ID_UC.isin(all_UCs.ID_UC)]
    print '# inspected ID_UCs: {0}'.format(len(inspected_UCs))

    uninspected_UCs = DataFrame(all_UCs[~all_UCs.ID_UC.isin(inspected_UCs.ID_UC)].ID_UC)
    print '# uninspected ID_UCs: {0}'.format(len(uninspected_UCs))

    # Add labels
    all_UCs['s'] = np.where(all_UCs.ID_UC.isin(inspected_UCs.ID_UC), 1, 0)

    return all_UCs


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

    clf = tree.DecisionTreeClassifier(class_weight='balanced')

    res = run_randomsearch(X, y, clf, param_dist, cv=10, n_iter_search=100)
    pickle.dump(res, open('bias_{0}.p'.format(name), 'wb'))


def main_single_feature():
    for name in FEATURES.keys():
        print 'Run experiments for feature {0}'.format(name)

        columns = FEATURES[name]['cols']

        # Remove ID_UC column
        data = get_labeled_data(columns).values[:, 1:]
        _, n = data.shape
        X = data[:, :n-1]
        y = data[:, n-1].astype(int)

        X = get_converted_feature(name, X)

        execute_learning(name, X, y)


def main_compound_feature():
    for compound_name in COMPOUND_FEATURES.keys():
        print 'Run experiments for feature {0}'.format(compound_name)
        relevant_features = COMPOUND_FEATURES[compound_name]

        # Collect all relevant columns
        columns = map(lambda x: FEATURES[x]['cols'], relevant_features)
        columns = reduce(lambda x, y: x + y, columns, [])

        # Remove ID_UC column
        data = get_labeled_data(columns).values[:, 1:]
        _, n = data.shape
        X_temp = data[:, :n-1]
        y = data[:, n-1].astype(int)

        X = []
        i = 0
        # Convert each feature
        for name in relevant_features:
            no_columns = len(FEATURES[name]['cols'])
            X_local = get_converted_feature(name, X_temp[:, i:i + no_columns])

            # Merge into X
            if i == 0:
                X = X_local
            else:
                X = np.concatenate((X, X_local), axis=1)

            i += no_columns
        print '{0} shape {1}'.format(compound_name, X.shape)
        print ''

        execute_learning('compound_{0}'.format(compound_name), X, y)


if __name__ == '__main__':
    main_single_feature()
