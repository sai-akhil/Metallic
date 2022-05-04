import sklearn
import evaluation
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK,Trials
from hyperopt.pyll.base import scope
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")

def decision_tree(X_train, y_train,X_test,y_test,state):
    # spl = ["best","random"]
    # cri = ["gini", "entropy"]

    # space = {
    #     "splitter": hp.choice("splitter", spl),
    #     "max_depth": hp.quniform("max_depth", 1, 7,1),
    #     "criterion": hp.choice("criterion", cri),
    # }

    # def hyperparameter_tuning(params):
    #     clf = DecisionTreeClassifier(**params)
    #     acc = cross_val_score(clf, X_train, y_train,scoring="f1_macro").mean()
    #     return {"loss": -acc, "status": STATUS_OK}

    # trials = Trials()

    # best = fmin(
    #     fn=hyperparameter_tuning,
    #     space = space, 
    #     algo=tpe.suggest, 
    #     max_evals=3, 
    #     trials=trials
    # )

    # # print("Best: {}".format(best))
    # x = cri[best.get("criterion")]
    # y = best.get("max_depth")
    # z = spl[best.get("splitter")]
    # print(x,y,z)

    a = time.time()
    dt = sklearn.tree.DecisionTreeClassifier().fit(X_train, y_train)
    b = time.time()
    train_time = (b - a)
    c = time.time()
    y_pred = dt.predict(X_test)
    d = time.time()
    predict_time = (d - c)

    lr_probs = dt.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]

    return evaluation.evaluation(y_pred, y_test, state, lr_probs, X_test)