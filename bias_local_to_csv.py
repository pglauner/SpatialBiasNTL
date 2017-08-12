import numpy as np
import pandas as pd

from sklearn.grid_search import _CVScoreTuple
import pickle
from get_data import get_raw_ID_UC
from config import FEATURES


def get_local_results(file_name):
    print 'Reading {0}'.format(file_name)
    return pickle.load(open('{0}.p'.format(file_name), 'rb'))


def get_median_location(UCs, locations, cols):
    locations_UCs = locations[locations['ID_UC'].isin(UCs)]
    return np.median(locations_UCs[cols], axis=0)


def get_centroids_and_values(dump):
    centroids = []
    values = []

    print 'Load all locations'
    location_cols = FEATURES['xy']['cols']
    locations = get_raw_ID_UC(location_cols)

    for i, key in enumerate(dump):
        print 'Compute centroid {0}/{1}'.format(i + 1, len(dump))
        UCs = dump[key]['ID_UCs']
        CV_result = dump[key]['res']

        if isinstance(CV_result, _CVScoreTuple):
            centroids.append(get_median_location(UCs, locations, location_cols))
            values.append(CV_result.mean_validation_score)

    return np.array(centroids), np.array(values)


def generate_csv(file_name):
    dump = get_local_results(file_name)
    centroids, values = get_centroids_and_values(dump)
    res = np.c_[centroids, values]
    df = pd.DataFrame(res)
    df.to_csv('{0}_dump.csv'.format(file_name), header=['longitude', 'latitude', 'bias'], index=False)


if __name__ == '__main__':
    for level in xrange(1, 5):
        generate_csv('local_res/bias_local_xy_level_{0}'.format(level))
