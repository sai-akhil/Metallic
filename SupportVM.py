import time
import evaluation
import sklearn


def SVM_classifier(X_train, y_train,X_test,y_test,state):
    a = time.time()
    SVM = sklearn.svm.SVC().fit(X_train, y_train)
    b = time.time()
    train_time=(b - a)
    c = time.time()
    y_pred = SVM.predict(X_test)
    d = time.time()
    predict_time=(d - c)
    return evaluation.evaluation(y_pred, y_test,state)