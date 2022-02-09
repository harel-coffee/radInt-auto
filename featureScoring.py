import numpy as np
import pandas as pd
from pymrmre import mrmr
import cv2

from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import RFE, RFECV
from ITMO_FS.filters.multivariate.FCBF import FCBFDiscreteFilter
from ITMO_FS.filters.univariate import chi2_measure
from ITMO_FS.filters.univariate import anova
from ITMO_FS.filters.univariate import information_gain
from ITMO_FS.filters.univariate import gini_index
from ITMO_FS.filters.univariate import su_measure
from ITMO_FS.filters import MCFS
from ITMO_FS.filters import UDFS
from ITMO_FS.filters.univariate import pearson_corr
from scipy.stats import kendalltau
from ITMO_FS.filters.univariate import laplacian_score
from ITMO_FS.filters.univariate import f_ratio_measure
from ITMO_FS.filters.univariate import fechner_corr
from ITMO_FS.filters.univariate import spearman_corr
from extraFeatureSelections import relief_measure, FCBFK
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import ICAP
from ITMO_FS.filters.multivariate import DCSF
from skfeature.function.information_theoretical_based import CIFE
from ITMO_FS.filters.multivariate import MIFS
from skfeature.function.information_theoretical_based import CMIM
from ITMO_FS.filters.multivariate import MRI
from skfeature.function.similarity_based import trace_ratio
from skfeature.function.statistical_based import t_score
from extraFeatureSelections import wilcoxon_score
from sklearn.utils import resample
import boruta


def boruta_fct (X, y):
    rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced_subsample')
    b = boruta.BorutaPy (rfc, n_estimators = nFeatures)
    b.fit(X, y)
    scores = np.max(b.ranking_) - b.ranking_
    return scores



def randlr_fct (X, y):
    # only 100 instead of 1000
    scores = None
    for k in range(25):
        boot = resample(range(0,X.shape[0]), replace=True, n_samples=X.shape[0], random_state=k)
        model = LogisticRegression(solver = 'lbfgs', random_state = k)
        model.fit(X[boot,:], y[boot])
        if scores is None:
            scores = model.coef_[0]*0
        scores = scores + np.abs(model.coef_[0])
    return scores



def variance (X, y):
    scores = np.var(X, axis = 0)
    return scores


def trace_ratio_score (X, y, nFeatures):
    fidx, fscore, _ = trace_ratio.trace_ratio (X,y, n_selected_features = nFeatures)
    scores = [0]*X.shape[1]
    for j in range(len(fidx)):
        scores[fidx[j]] = fscore[j]
    scores = np.asarray(scores, dtype = np.float32)
    return scores


def mrmr_score (X, y, nFeatures):
    Xp = pd.DataFrame(X, columns = range(X.shape[1]))
    yp = pd.DataFrame(y, columns=['Target'])

    # we need to pre-specify the max solution length...
    solutions = mrmr.mrmr_ensemble(features = Xp, targets = yp, solution_length=nFeatures, solution_count=1)
    scores = [0]*Xp.shape[1]
    for j,z in enumerate(solutions.iloc[0][0]):
        scores[z] = (len(solutions.iloc[0][0]) - j)/len(solutions.iloc[0][0])
    scores = np.asarray(scores, dtype = np.float32)
    return scores



def mri_score_fct (X, y):
    selected_features = []
    other_features = [i for i in range(0, X.shape[1]) if i not in selected_features]
    scores = MRI(np.array(selected_features), np.array(other_features), X, y)
    return scores



def cmim_score (X, y, nFeatures):
    sol, _, _ =CMIM.cmim (X,y, n_selected_features = nFeatures)
    scores = [0]*X.shape[1]
    for j,z in enumerate(sol):
        scores[z] = (len(sol) - j)/len(sol)
    scores = np.asarray(scores, dtype = np.float32)
    return scores



def mifs_score_fct (X, y):
    selected_features = []
    other_features = [i for i in range(0, X.shape[1]) if i not in selected_features]
    scores = MIFS(np.array(selected_features), np.array(other_features), X, y, beta = 0.5)
    return scores



def cife_score (X, y, nFeatures):
    sol, _, _ = CIFE.cife (X,y, n_selected_features = nFeatures)
    scores = [0]*X.shape[1]
    for j,z in enumerate(sol):
        scores[z] = (len(sol) - j)/len(sol)
    scores = np.asarray(scores, dtype = np.float32)
    return scores



def dcsf_score_fct (X, y):
    selected_features = []
    other_features = [i for i in range(0, X.shape[1]) if i not in selected_features]
    scores = DCSF(np.array(selected_features), np.array(other_features), X, y)
    return scores


def icap_score (X, y, nFeatures):
    sol, _, _ =ICAP.icap (X,y, n_selected_features = nFeatures)
    scores = [0]*X.shape[1]
    for j,z in enumerate(sol):
        scores[z] = (len(sol) - j)/len(sol)
    scores = np.asarray(scores, dtype = np.float32)
    return scores



def jmi_score (X, y, nFeatures):
    sol, _, _ = JMI.jmi (X,y, n_selected_features = nFeatures)
    scores = [0]*X.shape[1]
    for j,z in enumerate(sol):
        scores[z] = (len(sol) - j)/len(sol)
    scores = np.asarray(scores, dtype = np.float32)
    return scores



def laplacian_score_fct (X, y):
    scores = laplacian_score(X,y)
    return -scores


def kendall_corr_fct (X, y):
    scores = [0]*X.shape[1]
    for k in range(X.shape[1]):
        scores[k] = 1-kendalltau(X[:,k], y)[1]
    return np.array(scores)


def udfs_fct (X, y):
    udfs = UDFS(nFeatures)
    idxList = udfs.feature_ranking(X)
    scores = [1 if idx in idxList else 0 for idx in range(X.shape[1])]
    return np.array(scores)


def mcfs_fct (X, y):
    mcfs = MCFS(nFeatures, scheme='0-1') # dot is broken
    idxList = mcfs.feature_ranking(X)
    scores = [1 if idx in idxList else 0 for idx in range(X.shape[1])]
    return np.array(scores)


# this is possibly "broken" because it returnes always one feature, regardless
# of the dataset. the same is true for the implementation in skfeature
# but also for one https://github.com/mbt-ludovic-c/FCBF/blob/python3/src/fcbf.py
# but it seems normal-- the below FCBF returns 2 features, at times 1.
def fcbf_fct_broken (X, y):
    fcbf = FCBFDiscreteFilter()
    fcbf.fit(X,y)
    idxList = fcbf.selected_features
    scores = [1 if idx in idxList else 0 for idx in range(X.shape[1])]
    return np.array(scores)



def fcbf_score (X, y, nFeatures):
    fcbf = FCBFK(k = nFeatures)
    fcbf.fit(X, y)
    idxList = fcbf.idx_sel
    scores = [1 if idx in idxList else 0 for idx in range(X.shape[1])]
    return np.array(scores)



def mrmre_score (X, y, nFeatures):
    Xp = pd.DataFrame(X, columns = range(X.shape[1]))
    yp = pd.DataFrame(y, columns=['Target'])

    # we need to pre-specify the max solution length...
    solutions = mrmr.mrmr_ensemble(features = Xp, targets = yp, solution_length=nFeatures, solution_count=5)
    scores = [0]*Xp.shape[1]
    for k in solutions.iloc[0]:
        for j, z in enumerate(k):
            scores[z] = scores[z] + Xp.shape[1] - j
    scores = np.asarray(scores, dtype = np.float32)
    scores = scores/np.sum(scores)
    return scores


def svmrfe_score_fct (X, y):
    svc = LinearSVC (C=1)
    rfe = RFECV(estimator=svc, step=0.10, scoring='roc_auc', n_jobs=1)
    rfe.fit(X, y)
    scores = rfe.ranking_
    return scores


def bhattacharyya_score_fct (X, y):
    yn = y/np.sum(y)
    yn = np.asarray(yn, dtype = np.float32)
    scores = [0]*X.shape[1]
    for j in range(X.shape[1]):
        xn = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j] - np.min(X[:,j])))
        xn = xn/np.sum(xn)
        xn = np.asarray(xn, dtype = np.float32)
        scores[j] = cv2.compareHist(xn, yn, cv2.HISTCMP_BHATTACHARYYA)

    scores = np.asarray(scores, dtype = np.float32)
    # ties = {i:list(scores).count(i) for i in scores if list(scores).count(i) > 1}
    # print(ties)
    return -scores


def dummy_score (X, y):
    scores = np.ones(X.shape[1])
    return scores




#
