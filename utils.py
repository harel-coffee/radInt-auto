from sklearn.metrics import roc_curve, auc, roc_auc_score
from math import sqrt
import numpy as np
import scipy.stats
from scipy import stats
import shutil


# ties will get broken in a unspecified way--
# but we need ties to be broken the same all along.
# this is not so complicated:
# we always use the 'lower' features when breaking ties.
# so if scoring is (0, 5, 5, 0, 3, 2, 5) and we need to
# select 2 features, we choose f1,f2 and not f6.


from sklearn.feature_selection import f_classif
from sklearn.feature_selection._univariate_selection import _BaseFilter

class SelectKBestMY(_BaseFilter):
    """Select features according to the k highest scores.
    Read more in the :ref:`User Guide <univariate_feature_selection>`.
    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.
        .. versionadded:: 0.18
    k : int or "all", default=10
        Number of top features to select.
        The "all" option bypasses selection, for use in a parameter search.
    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.
    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned only scores.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    See Also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    mutual_info_regression: Mutual information for a continuous target.
    SelectPercentile: Select features based on percentile of the highest
        scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.
    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.
    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import SelectKBest, chi2
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
    >>> X_new.shape
    (1797, 20)
    """

    def __init__(self, score_func=f_classif, *, k=10):
        super().__init__(score_func=score_func)
        self.k = k

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be >=0, <= n_features = %d; got %r. "
                "Use k='all' to return all features." % (X.shape[1], self.k)
            )

    def _get_support_mask(self):
        check_is_fitted(self)

        if self.k == "all":
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.k == 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            mask = np.zeros(scores.shape, dtype=bool)

            # Request a stable sort. Mergesort takes more memory (~40MB per
            # megafeature on x86-64).

            scores = np.array([1,1,1,1,1,1,1,0,-1,-2,-3])
            np.random.shuffle(scores)
            ss = zip(scores, np.arange(scores.shape[0]))
            sorted(ss, key=lambda x: x[0])
            list(ss)
            np.argsort(ss)
            k = 4
            mask[np.argsort(scores, kind="mergesort")[-k :]] = 1
            mask

            mask[np.argsort(scores, kind="mergesort")[-self.k :]] = 1
            return mask



def recreatePath (path, create = True):
        print ("Recreating path ", path)
        try:
                shutil.rmtree (path)
        except:
                pass

        if create == True:
            try:
                    os.makedirs (path)
            except:
                    pass
        print ("Done.")



def getOptimalThreshold (fpr, tpr, threshold, verbose = False):
    # own way
    minDistance = 2
    bestPoint = (2,-1)
    bestThreshold = 0
    for i in range(len(fpr)):
        p = (fpr[i], tpr[i])
        d = sqrt ( (p[0] - 0)**2 + (p[1] - 1)**2 )
        if verbose == True:
            print (p, d)
        if d < minDistance:
            minDistance = d
            bestPoint = p
            bestThreshold = i

    return thres[bestThreshold]




def findOptimalCutoff (fpr, tpr, threshold, verbose = False):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    fpr, tpr, threshold

    Returns
    -------
    list type, with optimal cutoff value

    """

    # own way
    minDistance = 2
    bestPoint = (2,-1)
    for i in range(len(fpr)):
        p = (fpr[i], tpr[i])
        d = sqrt ( (p[0] - 0)**2 + (p[1] - 1)**2 )
        if verbose == True:
            print (p, d)
        if d < minDistance:
            minDistance = d
            bestPoint = p

    if verbose == True:
        print ("BEST")
        print (minDistance)
        print (bestPoint)
    sensitivity = bestPoint[1]
    specificity = 1 - bestPoint[0]
    return sensitivity, specificity


if __name__ == "__main__":
    y = np.asarray([1,0,0,0,1, 1,0,1,1,1, 1,0])
    A = np.asarray([0.4, 0.1, 0.2, 0.4, 0.2, 0.9, 0.7, 0.4, 0.1, 0.2, 0.9, 0.7])
    B = np.asarray([0.4, 0.3, 0.1, 0.2, 0.4, 0.2, 0.9, 0.7, 0.4, 0.1, 0.2, 0.4])
    p = testAUC (y, A, B)

#
