import sklearn
import imblearn
import numpy as np
def evaluation(y_pred,y_true,state):
    if state == 'multiclass':
        value='macro'
    else:
        value = 'binary'
    f1_score = sklearn.metrics.f1_score(y_true, y_pred,average=value)
    accuracy=sklearn.metrics.accuracy_score(y_true, y_pred)
    if state == 'multiclass':
        value='multiclass'
    else:
        value = 'binary'
    geometric_mean=imblearn.metrics.geometric_mean_score(y_true, y_pred,average=value) #geometric mean
    if state == 'multiclass':
        value='weighted'
    else:
        value = 'binary'
    precision=sklearn.metrics.precision_score(y_true, y_pred, average=value,labels=np.unique(y_pred))
    recall=sklearn.metrics.recall_score(y_true, y_pred,average=value)
    return f1_score,accuracy,geometric_mean,precision,recall





