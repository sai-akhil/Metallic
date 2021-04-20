import xgboost as xg
import numpy as np
import pandas as pd
import data_handling
import distance
import hypersphere
import overlapping,kmeans
from scipy.stats import rankdata
def sample(i):
    switcher = {
        1: 'None',
        2: 'SMOTE',
        3: 'NearMiss',
        4: 'SMOTEENN',
        5: 'Randomoversampling',
        6: 'ADASYN',
        7: 'BorderlineSMOTE',
        8: 'SVMSMOTE',
        9: 'RandomUnderSampler',
        10: 'ClusterCentroids',
        11: 'NearMissversion1',
        12: 'NearMissversion2',
        13: 'NearMissversion3',
        14: 'TomekLinks',
        15: 'EditedNearestNeighbours',
        16: 'RepeatedEditedNearestNeighbours',
        17: 'AllKNN',
        18: 'CondensedNearestNeighbour',
        19: 'NeighbourhoodCleaningRule',
        20: 'InstanceHardnessThreshold',
        21: 'SMOTETomek'
    }
    return switcher.get(i, "Invalid sampling method")
df = pd.read_csv('Metafeatures/features.csv')
filename='ecoli1.csv'#input("enter the file name:")
metrics='F1'#input("enter the metrics to be considered:")
print("select the Classifiers:")
print("1.KNN")
print("2.DT")
print("3.GNB")
print("4.SVM")
print("5.RF")
classifier=5#input("enter the classifier:")
rows=df[df["Classifier"]==classifier]
y_train=np.array(rows[metrics]) #y_train
x_train=rows.iloc[:,1:26] #Xtrain ready
X, y = data_handling.loading(filename)
no_of_rows_original = X.shape[0]
no_of_columns_original = X.shape[1]
no_of_class=len(np.unique(y))
if no_of_class > 2:
    state = 'multiclass'
    state_value = 1
else:
    state = 'binaryclass'
    state_value = 0
# unsupervised_kmeans
Silhouettescore, DaviesBouldinscore, Calinskiharabazscore, Cohesionscore, Separationscore, RMSSTDscore, RSscore, XBscore, Adjustedrandomscore, Adjusted_mutual_info_score, Fowlkes_mallows_score, Normalizedmutualinfoscore = kmeans.k_Means(X, y)
# imbalanced ratio
y = y.astype(int)
classes_data = list(np.unique(y))
list_of_instance = [sum(y == c) for c in classes_data]
min_instance = min(list_of_instance)
max_instance = max(list_of_instance)
imbalanced_ratio_before_sampling = min_instance / max_instance
# number of hyperspheres of minority and majority and avg class
minority_class = list_of_instance.index(min(list_of_instance))
majority_class = list_of_instance.index(max(list_of_instance))
hyperCentres = np.array(hypersphere.create_hypersphere(X, y))
distance_between_classes = distance.distance(hyperCentres, np.unique(y))
classes_hypersphere = list(set(hyperCentres[:, hyperCentres.shape[1] - 1]))
groupsPerClass = [sum(hyperCentres[:, hyperCentres.shape[1] - 1] == c) for c in classes_hypersphere]
minority_index = groupsPerClass.index(min(groupsPerClass))
majority_index = groupsPerClass.index(max(groupsPerClass))
minority_class_index = classes_hypersphere.index(minority_class)
majority_class_index = classes_hypersphere.index(majority_class)
minority_hypersphere = [minority_class, groupsPerClass[minority_class_index]]  # groupsPerClass[minority_class_index] will give you minority class hypersphere
majority_hypersphere = [majority_class, groupsPerClass[majority_class_index]]
total_hypersphere = sum(groupsPerClass)  # total number of hyperspheres
# samples per hypershere
total_number_instances = sum(list_of_instance)
samplesperhypersphere = total_number_instances / total_hypersphere
total_instance_minority = list_of_instance[minority_class]
total_instance_majority = list_of_instance[majority_class]
hypersphere_minority = groupsPerClass[minority_class_index]
hypersphere_majority = groupsPerClass[majority_class_index]
samplesperhypersphere_minority = total_instance_minority / hypersphere_minority
samplesperhypersphere_majority = total_instance_majority / hypersphere_majority

# volume of overlap
over=overlapping.volume_overlap(X,y)
test_x=[]
test_row=[no_of_rows_original,no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over]
for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22]:
    test_row.append(i)
    test_x.append(test_row)
    test_row=test_row[:24]
model = xg.XGBRegressor(colsample_bytree=0.4,gamma=0,learning_rate=0.07,max_depth=3,min_child_weight=1.5,n_estimators=10000,reg_alpha=0.75,reg_lambda=0.45,subsample=0.6,seed=42)
x_train=np.array(x_train)
model.fit(np.array(x_train),y_train)
y_pred=model.predict(np.array(test_x))
#ranking=rankdata(y_pred,)
y_pred=pd.DataFrame(y_pred)
ranking=y_pred.rank(ascending=False)
ranking=np.array(ranking)
list_rank=[]
for i in ranking:
    for j in i:
        list_rank.append(float(j))
sort_ranking=list_rank.copy()
sort_ranking.sort()
top_3=sort_ranking[:3]
index_list=[]
for i in top_3:
    index_list.append(list_rank.index(i))
print(index_list)
first=sample(index_list[0]+1)
second=sample(index_list[1]+1)
third=sample(index_list[2]+1)
print("The recommended sampling methods are:")
print("1.",first)
print("2.",second)
print("3.",third)


#new_rows=np.array(new_rows)