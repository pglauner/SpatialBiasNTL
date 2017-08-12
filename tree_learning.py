# Inspired by http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html

from time import time
from operator import itemgetter
import numpy as np
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer


def absolute_matthews_corrcoef(y_true, y_pred, sample_weight=None):
        return np.abs(matthews_corrcoef(y_true, y_pred))


def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0]


def run_randomsearch(X, y, clf, param_dist, cv=5,
                     n_iter_search=20,
                     scoring=make_scorer(absolute_matthews_corrcoef)):
    """Run a random search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_dist -- [dict] list, distributions of parameters
                  to sample
    cv -- fold of cross-validation, default 5
    n_iter_search -- number of random parameter sets to try,
                     default 20.

    Returns
    -------
    top_params -- [dict] from report()
    """
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, random_state=42,
                                       scoring=scoring, n_jobs=-1, verbose=10, cv=cv)

    start = time()
    random_search.fit(X, y)
    print(("\nRandomizedSearchCV took {:.2f} seconds "
           "for {:d} candidates parameter "
           "settings.").format((time() - start),
                               n_iter_search))

    top_scores = report(random_search.grid_scores_, 3)
    return top_scores
