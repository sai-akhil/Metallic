import time
import evaluation
from sklearn.neighbors import KNeighborsClassifier


def knn_classifier(X_train, y_train,X_test,y_test,state):
    a = time.time()
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    b = time.time()
    train_time=(b - a)
    c = time.time()
    y_pred = knn.predict(X_test)
    d = time.time()
    predict_time=(d - c)
    return evaluation.evaluation(y_pred, y_test,state)