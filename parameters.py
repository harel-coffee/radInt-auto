from collections import OrderedDict
import numpy as np


### parameters
TrackingPath = "./results/mlrun"
nCV = 10
DPI = 300


# datasets
dList =  [ "Arita2018",  "Carvalho2018", \
                "Hosny2018A", "Hosny2018B", "Hosny2018C", \
                "Ramella2018",  "Lu2019","Sasaki2019", "Toivonen2019", "Keek2020", "Li2020", \
                "Park2020", "Song2020", "Veeraraghavan2020" ]



fselParameters = OrderedDict({
    # these are 'one-of'
    "FeatureSelection": {
        "N": [1,2,4,8,16,32,64],
        "Methods": {
            "MIM": {},
            "LASSO": {"C": [1.0]},
            "Anova": {},
            "Bhattacharyya": {},
            "FCBF": {},
            "MRMRe": {},
            "ET": {},
            "Kendall": {},
        }
    }
})

clfParameters = OrderedDict({
    "Classification": {
        "Methods": {
            "LDA": {},
            "SVM": {"C":np.logspace(-6, 6, 7, base = 2.0), "gamma":["auto"]},
            "RBFSVM": {"C":np.logspace(-6, 6, 7, base = 2.0), "gamma":["auto"]},
            "RandomForest": {"n_estimators": [50, 250, 500]},
            "XGBoost": {"learning_rate": [0.001, 0.1, 0.3, 0.9], "n_estimators": [50, 250, 500]},
            "LogisticRegression": {"C": np.logspace(-6, 6, 7, base = 2.0) },
            "NeuralNetwork": {"layer_1": [4, 16, 64], "layer_2": [4, 16, 64], "layer_3": [4, 16, 64]},
            "NaiveBayes": {}
        }
    }
})



#
