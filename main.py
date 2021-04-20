#get the dataset from files
import csv,overlapping
import distance
from openpyxl import load_workbook
import numpy as np
import pandas as pd
import math
import glob,kmeans

import hypersphere
import knn,decision_tree,Gaussian,SupportVM,random_forest
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
import sklearn
import time
import os
from os import path

from sklearn.neighbors import KNeighborsClassifier

import data_handling,missing_values
from sklearn.model_selection import StratifiedKFold


def handlingMissingValues(X,y,data_clean):

    option = data_clean

    if(option == 1):
        rows,cols = X.shape
        meansArray = []
        class_label_mapper = {}
        index =0
        for i in np.unique(y):
            class_label_mapper[i] = index
            index+=1

        for i in range(len(np.unique(y))):
            meansArray.append([])
            for j in range(X.shape[1]):
                meansArray[i].append(np.nanmean(X[tuple(list(np.where(y == i)))][:, j]))
        for i in range(rows):
            for j in range(cols):
                if(math.isnan(X[i][j])):
                    if(math.isnan(meansArray[class_label_mapper[y[i]]][j])):
                        X[i][j] = 0
                    else:
                        X[i][j] = meansArray[class_label_mapper[y[i]]][j]
        return X

    elif(option == 2):
        X = X[~np.isnan(X).any(axis=1)]
        return X
    elif(option ==3):
        X = X[:,~np.isnan(X).any(axis=0)]
        return X
    else:
        print("Invalid data cleaning option\n")
p ='Metafeatures/features.csv'
l = path.exists(p)
fields=['Dataset', 'Original Rows', 'Columns','Type','Silhouettescore','DaviesBouldinscore','Calinskiharabazscore','Cohesionscore','Separationscore','RMSSTDscore','RSscore','XBscore','Adjustedrandomscore','Adjusted_mutual_info_score','Fowlkes_mallows_score','Normalizedmutualinfoscore','imbalanced_ratio_before_sampling','total_hypersphere','hypersphere_minority','hypersphere_majority','samplesperhypersphere','samplesperhypersphere_minority','samplesperhypersphere_majority','Average distance between class','overlapping','Sampling','Classifier','F1','G-mean','Accuracy','Precision','Recall']
if l == True:
    os.remove(p)
write_file=open(p, 'w',newline='')
csvwriter = csv.writer(write_file)
csvwriter.writerow(fields)
#csvwriter.writerow(["SN", "Name", "Contribution"])          to add each rows
#write_file.close()
#exit(0)
files = glob.glob("C:/Users/DELL/PycharmProjects/metalearning_version1/datasets/*.csv")
spl_word = "\\"
train_time=[]
predict_time=[]
for file in files:
    file_name = file.partition(spl_word)[2]
    print("file name:", file_name)
    X, y = data_handling.loading(file_name)
    X = missing_values.handlingMissingValues(X, y, 1)
    no_of_rows_original = X.shape[0]
    no_of_columns_original = X.shape[1]
    no_of_class=len(np.unique(y))
    if no_of_class > 2:
        state='multiclass'
        state_value=1
    else:
        state='binaryclass'
        state_value = 0
    # unsupervised_kmeans
    Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore=kmeans.k_Means(X,y)
    #imbalanced ratio
    y=y.astype(int)
    classes_data=list(np.unique(y))
    list_of_instance = [sum(y == c) for c in classes_data]
    min_instance=min(list_of_instance)
    max_instance = max(list_of_instance)
    imbalanced_ratio_before_sampling=min_instance/max_instance
    # number of hyperspheres of minority and majority and avg class
    minority_class=list_of_instance.index(min(list_of_instance))
    majority_class = list_of_instance.index(max(list_of_instance))
    hyperCentres=np.array(hypersphere.create_hypersphere(X,y))
    distance_between_classes=distance.distance(hyperCentres,np.unique(y))
    classes_hypersphere = list(set(hyperCentres[:, hyperCentres.shape[1] - 1]))
    groupsPerClass = [sum(hyperCentres[:, hyperCentres.shape[1] - 1] == c) for c in classes_hypersphere]
    minority_index = groupsPerClass.index(min(groupsPerClass))
    majority_index = groupsPerClass.index(max(groupsPerClass))
    minority_class_index = classes_hypersphere.index(minority_class)
    majority_class_index=classes_hypersphere.index(majority_class)
    minority_hypersphere = [minority_class, groupsPerClass[minority_class_index]] #groupsPerClass[minority_class_index] will give you minority class hypersphere
    majority_hypersphere = [majority_class, groupsPerClass[majority_class_index]]
    total_hypersphere = sum(groupsPerClass)#total number of hyperspheres
    #samples per hypershere
    total_number_instances=sum(list_of_instance)
    samplesperhypersphere=total_number_instances/total_hypersphere
    total_instance_minority=list_of_instance[minority_class]
    total_instance_majority = list_of_instance[majority_class]
    hypersphere_minority=groupsPerClass[minority_class_index]
    hypersphere_majority=groupsPerClass[majority_class_index]
    samplesperhypersphere_minority=total_instance_minority/hypersphere_minority
    samplesperhypersphere_majority=total_instance_majority/hypersphere_majority


    # volume of overlap
    over=overlapping.volume_overlap(X,y)

    #d = handlingMissingValues(d, ts, 1)
    crossvalidation=5
    kf = StratifiedKFold(n_splits=crossvalidation) #cross validation to 5
    f1_knn=[]
    acc_knn=[]
    gmean_knn=[]
    f1_dt = []
    acc_dt = []
    gmean_dt = []
    f1_GNB = []
    acc_GNB = []
    gmean_GNB = []
    f1_SVM = []
    acc_SVM = []
    gmean_SVM = []
    f1_RF = []
    acc_RF = []
    gmean_RF = []
    preci_knn=[]
    preci_dt=[]
    preci_SVM=[]
    preci_GNB=[]
    preci_RF=[]
    rec_knn = []
    rec_dt = []
    rec_SVM = []
    rec_GNB = []
    rec_RF = []


    print("KNN classifier")
    print("SVM classifier")
    print("Decision Tree")
    print("")
    for i in ['None','SMOTE','NearMiss','SMOTEENN','Randomoversampling','ADASYN','BorderlineSMOTE','SVMSMOTE','RandomUnderSampler','ClusterCentroids','NearMissversion1','NearMissversion2','NearMissversion3','TomekLinks','EditedNearestNeighbours','RepeatedEditedNearestNeighbours','AllKNN','CondensedNearestNeighbour','NeighbourhoodCleaningRule','InstanceHardnessThreshold','SMOTETomek']:
        if i == 'None':
            uX, uy = X,y
            ii=1
        elif i == 'SMOTE':
            try:
                # a = np.bincount(y)
                smt = SMOTE()
                uX, uy = smt.fit_sample(X, y)
                ii=2
                # b = np.bincount(uy)
            except:
                continue
        elif i == 'NearMiss':
            try:
                nr = NearMiss()
                uX, uy = nr.fit_sample(X, y)
                ii=3
            except:
                continue
        elif i == 'SMOTEENN':
            try:
                sme = SMOTEENN(random_state=42)
                uX, uy = sme.fit_resample(X, y)
                ii=4
            except:
                continue
        elif i=='Randomoversampling':
            try:
                ros = RandomOverSampler(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ii=5
            except:
                continue
        elif i=='ADASYN':
            try:
                ros = ADASYN(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ii = 6
            except:
                continue
        elif i=='BorderlineSMOTE':
            try:
                ros = BorderlineSMOTE(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ii = 7
            except:
                continue
        elif i=='SVMSMOTE':
            try:
                ros = SVMSMOTE(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ii = 8
            except:
                continue
        elif i=='RandomUnderSampler':
            try:
                ros = RandomUnderSampler(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ii = 9
            except:
                continue
        elif i=='ClusterCentroids':
            try:
                ros = ClusterCentroids(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ii = 10
            except:
                continue
        elif i=='NearMissversion1':
            try:
                ros = NearMiss(version=1)
                uX, uy = ros.fit_resample(X, y)
                ii=11
            except:
                continue
        elif i=='NearMissversion2':
            try:
                ros = NearMiss(version=2)
                uX, uy = ros.fit_resample(X, y)
                ii = 12
            except:
                continue
        elif i=='NearMissversion3':
            try:
                ros = NearMiss(version=3)
                uX, uy = ros.fit_resample(X, y)
                ii=13
            except:
                continue
        elif i=='TomekLinks':
            try:
                ros = TomekLinks()
                uX, uy = ros.fit_resample(X, y)
                ii=14
            except:
                continue
        elif i=='EditedNearestNeighbours':
            try:
                ros = EditedNearestNeighbours()
                uX, uy = ros.fit_resample(X, y)
                ii = 15
            except:
                continue
        elif i=='RepeatedEditedNearestNeighbours':
            try:
                ros = RepeatedEditedNearestNeighbours()
                uX, uy = ros.fit_resample(X, y)
                ii = 16
            except:
                continue
        elif i=='AllKNN':
            try:
                ros = AllKNN()
                uX, uy = ros.fit_resample(X, y)
                ii = 17
            except:
                continue
        elif i=='CondensedNearestNeighbour':
            try:
                ros = CondensedNearestNeighbour(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ii=18
            except:
                continue
        elif i=='NeighbourhoodCleaningRule':
            try:
                ros = NeighbourhoodCleaningRule()
                uX, uy = ros.fit_resample(X, y)
                ii = 19
            except:
                continue
        elif i=='InstanceHardnessThreshold':
            try:
                ros = InstanceHardnessThreshold(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ii = 20
            except:
                continue
        elif i=='SMOTETomek':
            try:
                ros = SMOTETomek(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ii = 21
            except:
                continue
        no_of_rows_after = uX.shape[0]
        no_of_columns_after = uX.shape[1]
        after_sampling=np.bincount(uy)
        imbalanced_ratio_after_sampling=min(after_sampling)/max(after_sampling)
        for train_index, test_index in kf.split(uX, uy):
            # Splitting out training and testing data
            X_train, X_test = uX[train_index], uX[test_index]
            y_train, y_test = uy[train_index], uy[test_index]
            # knn
            f1_score_knn, accuracy_knn, geometric_mean_knn,precision_knn,recall_knn= knn.knn_classifier(X_train, y_train, X_test, y_test, state)
            f1_knn.append(f1_score_knn)
            acc_knn.append(accuracy_knn)
            gmean_knn.append(geometric_mean_knn)
            preci_knn.append(precision_knn)
            rec_knn.append(recall_knn)
            # decision tree
            f1_score_dt, accuracy_dt, geometric_mean_dt,precision_dt,recall_dt = decision_tree.decision_tree(X_train, y_train, X_test, y_test,
                                                                                  state)
            f1_dt.append(f1_score_dt)
            acc_dt.append(accuracy_dt)
            gmean_dt.append(geometric_mean_dt)
            preci_dt.append(precision_dt)
            rec_dt.append(recall_dt)
            # GaussianNB
            f1_score_GNB, accuracy_GNB, geometric_mean_GNB,precision_GNB,recall_GNB  = Gaussian.Gaussian(X_train, y_train, X_test, y_test, state)
            f1_GNB.append(f1_score_GNB)
            acc_GNB.append(accuracy_GNB)
            gmean_GNB.append(geometric_mean_GNB)
            preci_GNB.append(precision_GNB)
            rec_GNB.append(recall_GNB)
            # Support Vector Machine
            f1_score_SVM, accuracy_SVM, geometric_mean_SVM,precision_SVM,recall_SVM = SupportVM.SVM_classifier(X_train, y_train, X_test, y_test,
                                                                                      state)
            f1_SVM.append(f1_score_SVM)
            acc_SVM.append(accuracy_SVM)
            gmean_SVM.append(geometric_mean_SVM)
            preci_SVM.append(precision_SVM)
            rec_SVM.append(recall_SVM)
            # Random forest
            f1_score_RF, accuracy_RF, geometric_mean_RF,precision_RF,recall_RF  = random_forest.RF_classifier(X_train, y_train, X_test, y_test,
                                                                                      state)
            f1_RF.append(f1_score_RF)
            acc_RF.append(accuracy_RF)
            gmean_RF.append(geometric_mean_RF)
            preci_RF.append(precision_RF)
            rec_RF.append(recall_RF)
        # knn
        avg_f1_score_knn = sum(f1_knn) / crossvalidation
        avg_acc_knn = sum(acc_knn) / crossvalidation
        avg_gmean_knn = sum(gmean_knn) / crossvalidation
        avg_precision_knn=sum(preci_knn)/crossvalidation
        avg_recall_knn=sum(rec_knn)/crossvalidation
        # decision tree
        avg_f1_score_dt = sum(f1_dt) / crossvalidation
        avg_acc_dt = sum(acc_dt) / crossvalidation
        avg_gmean_dt = sum(gmean_dt) / crossvalidation
        avg_precision_dt = sum(preci_dt) / crossvalidation
        avg_recall_dt = sum(rec_dt) / crossvalidation

        # GaussianNB
        avg_f1_score_GNB = sum(f1_GNB) / crossvalidation
        avg_acc_GNB = sum(acc_GNB) / crossvalidation
        avg_gmean_GNB = sum(gmean_GNB) / crossvalidation
        avg_precision_GNB = sum(preci_GNB) / crossvalidation
        avg_recall_GNB = sum(rec_GNB) / crossvalidation
        # SVM
        avg_f1_score_SVM = sum(f1_SVM) / crossvalidation
        avg_acc_SVM = sum(acc_SVM) / crossvalidation
        avg_gmean_SVM = sum(gmean_SVM) / crossvalidation
        avg_precision_SVM = sum(preci_SVM) / crossvalidation
        avg_recall_SVM = sum(rec_SVM) / crossvalidation
        # RF
        avg_f1_score_RF = sum(f1_RF) / crossvalidation
        avg_acc_RF = sum(acc_RF) / crossvalidation
        avg_gmean_RF = sum(gmean_RF) / crossvalidation
        avg_precision_RF = sum(preci_RF) / crossvalidation
        avg_recall_RF = sum(rec_RF) / crossvalidation

        csvwriter.writerow([file_name, no_of_rows_original,no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over,ii,1,avg_f1_score_knn,avg_gmean_knn,avg_acc_knn,avg_precision_knn,avg_recall_knn])
        csvwriter.writerow([file_name, no_of_rows_original, no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over, ii, 2, avg_f1_score_dt,avg_gmean_dt, avg_acc_dt,avg_precision_dt,avg_recall_dt])
        csvwriter.writerow([file_name, no_of_rows_original, no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over, ii, 3,avg_f1_score_GNB,avg_gmean_GNB, avg_acc_GNB,avg_precision_GNB,avg_recall_GNB])
        csvwriter.writerow([file_name, no_of_rows_original, no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over, ii, 4,avg_f1_score_SVM,avg_gmean_SVM, avg_acc_SVM,avg_precision_SVM,avg_recall_SVM])
        csvwriter.writerow([file_name, no_of_rows_original, no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over, ii, 5,avg_f1_score_RF,avg_gmean_RF, avg_acc_RF,avg_precision_RF,avg_recall_RF])

        f1_knn=[]
        acc_knn=[]
        gmean_knn=[]
        preci_knn=[]
        rec_knn=[]
        f1_dt = []
        acc_dt = []
        gmean_dt = []
        preci_dt = []
        rec_dt = []
        f1_GNB = []
        acc_GNB  = []
        gmean_GNB  = []
        preci_GNB  = []
        rec_GNB  = []
        f1_SVM = []
        acc_SVM  = []
        gmean_SVM = []
        preci_SVM  = []
        rec_SVM  = []
        f1_RF = []
        acc_RF = []
        gmean_RF = []
        preci_RF = []
        rec_RF = []
write_file.close()










