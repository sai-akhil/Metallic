import pandas as pd
import data_handling
import numpy as np

import hypersphere,distance
import kmeans
import overlapping


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
df = pd.read_csv ('Metafeatures/features.csv')
rows=df.iloc[:,:25]
new_rows=rows.drop_duplicates(subset ="Dataset")
#metrics='F1'
#classifier='1'
#filename='ecoli1.csv'
filename='zoo.csv' #input("enter the file name:")
metrics="F1"#input("enter the metrics to be considered:")
print("select the Classifiers:")
print("1.KNN")
print("2.DT")
print("3.GNB")
print("4.SVM")
print("5.RF")
classifier=1 #input("enter the classifier:")
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
test_row=[filename,no_of_rows_original,no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over]
new_rows=np.array(new_rows)
filename_only=new_rows[:,0]
distance_test=[]
for i in new_rows:
    dist = np.linalg.norm(i[1:] - test_row[1:])
    distance_test.append(dist)
sorted_distance=distance_test.copy()
sorted_distance.sort()
nearest_neighbor=sorted_distance[:3]
nearest_dataset=[]
rank=[]
for i in nearest_neighbor:
    index=distance_test.index(i)
    dataset_name=filename_only[index]
    nearest_dataset.append(dataset_name)
for i in nearest_dataset:
    subset=df[df["Dataset"]==i]
    subsub=subset[subset["Classifier"]==int(classifier)]
    f1_score=subsub[metrics]
    sampling=subsub["Sampling"]
    rank.append(f1_score.rank(ascending=False))
rank0=np.array(rank[0])
rank1=np.array(rank[1])
rank2=np.array(rank[2])
sampling=np.array(sampling)
total_rank=rank0+rank1+rank2
copy_total_rank=total_rank.copy()
copy_total_rank.sort()
top_3=copy_total_rank[:3]
best=[]
total_rank=list(total_rank)
for i in top_3:
    index=total_rank.index(i)
    best.append(sampling[index])
first=best[0]
second=best[1]
third=best[2]
first_algorithm=sample(first)
second_algorithm=sample(second)
third_algorithm=sample(third)
print("The Recommended Sampling techniques:")
print("1.",first_algorithm)
print("2.",second_algorithm)
print("3.",third_algorithm)







