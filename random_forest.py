import time
import evaluation
import sklearn


def RF_classifier(X_train, y_train, X_test, y_test, state):
    a = time.time()
    rf = sklearn.ensemble.RandomForestClassifier().fit(X_train, y_train)
    b = time.time()
    train_time = (b - a)
    c = time.time()
    y_pred = rf.predict(X_test)
    d = time.time()
    predict_time = (d - c)
    return evaluation.evaluation(y_pred, y_test, state)


