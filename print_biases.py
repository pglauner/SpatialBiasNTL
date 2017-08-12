import pickle
from sklearn.grid_search import RandomizedSearchCV
from os import listdir
from os.path import isfile, join
from config import FEATURES
import numpy as np

# TODO: Change to your path
mypath = ''


def print_details(name):
    res = pickle.load(open(name, 'rb'))
    print 'Feature: {0}'.format(name)
    print 'Mean: {0}'.format(res.mean_validation_score)
    print 'Std: {0}'.format(np.std(res.cv_validation_scores))
    print ''


def main():
    feature_files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.p')
                        and not f.startswith('bias_compound')]

    print 'FEATURES'
    for name in feature_files:
        print_details(name)

    compound_feature_files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.p')
                                and f.startswith('bias_compound')]

    print ''
    print 'COMPOUND FEATURES'
    for name in compound_feature_files:
        print_details(name)


if __name__ == '__main__':
    main()
