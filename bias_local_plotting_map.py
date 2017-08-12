from cartopy.feature import NaturalEarthFeature, BORDERS, COASTLINE, OCEAN
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
from scipy.interpolate import griddata
from sklearn.grid_search import _CVScoreTuple
import seaborn as sns
import pickle
from get_data import get_raw_ID_UC
from config import FEATURES


# TODO: change to your coordinates
LON_L, LON_U, LAT_L, LAT_U = 20, 30, 40, 50


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


def plot_brazil_states(centroid, values, interpolation_method, grid_dim):
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(projection=ccrs.PlateCarree()))

    ax.set_extent([LON_L, LON_U, LAT_L, LAT_U])
    ax.stock_img()
    ax.add_feature(BORDERS)
    ax.add_feature(COASTLINE, linewidth=1)
    ax.add_feature(OCEAN)

    grid_x, grid_y = np.mgrid[LON_L:LON_U:grid_dim, LAT_L:LAT_U:grid_dim]
    grid_z = griddata(centroid, values, (grid_x, grid_y), method=interpolation_method, fill_value=0)
    # Do not allow negative score
    grid_z = np.where(grid_z > 0, grid_z, 0)

    print np.max(grid_z)

    #cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.contourf(grid_x, grid_y, grid_z, 100, transform=ccrs.PlateCarree(), cmap="Reds")
    plt.colorbar(orientation="horizontal")

    return fig, ax


def generate_plot(file_name, interpolation_methods, grid_dims):
    dump = get_local_results(file_name)
    centroids, values = get_centroids_and_values(dump)

    for interpolation_method in interpolation_methods:
        for grid_dim in grid_dims:
            print 'For {0}, {1} interpolation'.format(interpolation_method, grid_dim)
            fig, ax = plot_brazil_states(centroids, values, interpolation_method, grid_dim)
            states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                                     name='admin_1_states_provinces_shp', linewidth=1)
            _ = ax.add_feature(states, edgecolor='black')

            plt.tight_layout()
            #plt.show()
            plt.savefig('{0}_plot_{1}_{2}_f.eps'.format(file_name, interpolation_method, grid_dim), format='eps')
3

if __name__ == '__main__':
    for level in xrange(1, 5):
        generate_plot('local_res/bias_local_xy_level_{0}'.format(level), ['nearest', 'linear'], [1000j, 2000j])
