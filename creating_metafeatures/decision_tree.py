import sklearn
import evaluation
import time


def decision_tree(X_train, y_train,X_test,y_test,state):
    a = time.time()
    dt = sklearn.tree.DecisionTreeClassifier().fit(X_train, y_train)
    b = time.time()
    train_time = (b - a)
    c = time.time()
    y_pred = dt.predict(X_test)
    d = time.time()
    predict_time = (d - c)
    return evaluation.evaluation(y_pred, y_test, state)