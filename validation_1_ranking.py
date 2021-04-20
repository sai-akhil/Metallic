import pandas as pd
import numpy as np
import xgboost as xg
from sklearn import metrics
import os
from os import path
import csv
import sklearn
import krippendorff

def sample(i):
    switcher = {
        1: 'KNN',
        2: 'DT',
        3: 'GNB',
        4: 'SVM',
        5: 'RF'
    }
    return switcher.get(i, "Invalid sampling method")

p = 'Metafeatures/kappa_score1.csv'
p1='Metafeatures/krippendorf1.csv'
p2='Metafeatures/precission1.csv'
l = path.exists(p)
l1=path.exists(p1)
l2=path.exists(p2)
fields = ['Test_dataset', 'Classifier', 'F1', 'G-mean', 'Accuracy', 'Precision', 'Recall']
if l == True:
    os.remove(p)
if l1 == True:
    os.remove(p1)
if l2 == True:
    os.remove(p2)
write_file = open(p, 'w', newline='')
write_file1=open(p1,'w',newline='')
write_file2=open(p2,'w',newline='')
csvwriter = csv.writer(write_file)
csvwriter1 = csv.writer(write_file1)
csvwriter2 = csv.writer(write_file2)
csvwriter.writerow(fields)
csvwriter1.writerow(fields)
csvwriter2.writerow(fields)
df = pd.read_csv('Metafeatures/features_simple.csv')
rows=df.iloc[:,:25]
filename_only=list(np.unique(df['Dataset']))
new_rows=rows.drop_duplicates(subset ="Dataset")
for classifier in [1,2,3,4,5]:
    for i in filename_only:
        kapa_score = []
        precisson_5 = []
        krippen_score = []
        train = np.array(new_rows[new_rows['Dataset'] != i])
        test = np.array((new_rows[new_rows['Dataset'] == i]))
        test_dataset = test[0][0]
        for jj in test:
            test_row = list(jj)
        distance_test = []
        for i in train:
            dist = np.linalg.norm(i[1:] - np.array(test_row[1:]))
            distance_test.append(dist)
        sorted_distance = distance_test.copy()
        sorted_distance.sort()
        nearest_neighbor = sorted_distance[:3]
        nearest_dataset = []
        rank = []
        files = filename_only.copy()
        files.remove(test[0][0])
        for j in nearest_neighbor:
            index = distance_test.index(j)
            dataset_name = files[index]
            nearest_dataset.append(dataset_name)
        for metric in ['F1','G-mean','Accuracy','Precision','Recall']:
            average_y_pred = []
            for j in nearest_dataset:
                subset = df[df["Dataset"] == j]
                subsub = subset[subset["Classifier"] == int(classifier)]
                f1_score = list(subsub[metric])
                sampling = np.array(subsub["Sampling"])
                # find missing sampling
                missing_values=[]
                for jj in range(1,22):
                    if jj not in sampling:
                        missing_values.append(jj)
                for jj in missing_values:
                    f1_score.insert(int(jj)-1,0)
                f1_score=np.array(f1_score)
                average_y_pred.append(list(f1_score))
                f1_score=pd.DataFrame(f1_score)
                f1_score=f1_score.rank(ascending=False, method="first")
                rank.append(f1_score[0])
            total_y_pred = np.array(average_y_pred[0]) + np.array(average_y_pred[1]) + np.array(average_y_pred[2])
            y_pred = total_y_pred / 3
            rank0 = np.array(rank[0])
            rank1 = np.array(rank[1])
            rank2 = np.array(rank[2])
            sampling = [kk for kk in range(1,22)]
            total_rank = rank0 + rank1 + rank2
            copy_total_rank = total_rank.copy()
            #y_pred=copy_total_rank/3
            copy=pd.DataFrame(copy_total_rank)
            pred_rank=copy.rank(method="first")
            pred_rank=np.array(pred_rank)
            pred_rank_final=[]
            for kk in pred_rank:
                for jj in kk:
                    pred_rank_final.append(jj) # required to measure
            selected_rows=df[df['Dataset']==test_dataset]
            selected_rows_classifier=selected_rows[selected_rows['Classifier']==classifier]
            metric_selected=np.array(selected_rows_classifier[metric])
            metric_selected=list(metric_selected)
            selected_sampling_test=np.array(selected_rows_classifier['Sampling'])
            not_present=[]
            for kk in range(1,22):
                if kk not in selected_sampling_test:
                    not_present.append(kk)
            for kk in not_present:
                metric_selected.insert(int(kk)-1,0)
            y_test=metric_selected
            metric_selected=pd.DataFrame(metric_selected)
            real_ranking = metric_selected.rank(ascending=False, method="first")
            real_ranking=np.array(real_ranking)
            real_ranking_final=[]
            for kk in real_ranking:
                for jj in kk:
                    real_ranking_final.append(jj)
            kapa_score.append(sklearn.metrics.cohen_kappa_score(np.array(real_ranking_final), np.array(pred_rank_final)))
            reliability_data = [list(real_ranking_final), list(pred_rank_final)]
            krippen_score.append(krippendorff.alpha(reliability_data=reliability_data))
            real_5 = []
            pred_5 = []
            for r in range(1, 6):
                if r in real_ranking_final:
                    index = real_ranking_final.index(r)
                    real_5.append(index + 1)
                if r in pred_rank_final:
                    index1 = pred_rank_final.index(r)
                    pred_5.append(index1 + 1)
            count = 0
            for rk in pred_5:
                if rk in real_5:
                    count = count + 1
            precisson_5.append(count / 5)
            sampling_tech=['None', 'SMOTE', 'NearMiss', 'SMOTEENN', 'Randomoversampling', 'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE',
     'RandomUnderSampler', 'ClusterCentroids', 'NearMissversion1', 'NearMissversion2', 'NearMissversion3', 'TomekLinks',
     'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours', 'AllKNN', 'CondensedNearestNeighbour',
     'NeighbourhoodCleaningRule', 'InstanceHardnessThreshold', 'SMOTETomek']
            data = {'Dataset': [test_dataset] * 21, 'Classifier': [sample(classifier)] * 21, 'Metric': [metric] * 21,'predicted_score': list(y_pred),'predicted_ranking': list(pred_rank_final),'original_score': list(y_test), 'original_ranking': list(real_ranking_final),'Sampling':sampling_tech} #'predicted_score': list(y_pred)
            new_file = pd.DataFrame(data)
            path = 'test_file_result/' + sample(classifier) + '_' + metric + '_' + test_dataset
            print(path)
            new_file.to_csv(path, index=False)
            data = {}
        csvwriter.writerow([test_dataset, sample(classifier), kapa_score[0], kapa_score[1], kapa_score[2], kapa_score[3], kapa_score[4]])
        csvwriter1.writerow([test_dataset, sample(classifier), krippen_score[0], krippen_score[1], krippen_score[2], krippen_score[3], krippen_score[4]])
        csvwriter2.writerow([test_dataset, sample(classifier), precisson_5[0], precisson_5[1], precisson_5[2], precisson_5[3], precisson_5[4]])