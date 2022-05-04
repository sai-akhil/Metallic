#get the dataset from files
import csv,overlapping
import sys
sys.path.insert(1, "/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Recommendation_system")
import distance
from openpyxl import load_workbook
import numpy as np
import pandas as pd
import math
import glob,kmeans
from sklearn.preprocessing import LabelEncoder

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

import data_handling
from sklearn.model_selection import StratifiedKFold
import missing_values

import complexity_metric, wCM, dwCM

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
p ="/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/features.csv"
l = path.exists(p)
fields=['Dataset', 'Original Rows', 'Columns','Type','Silhouettescore','DaviesBouldinscore','Calinskiharabazscore','Cohesionscore','Separationscore','RMSSTDscore','RSscore','XBscore','Adjustedrandomscore','Adjusted_mutual_info_score','Fowlkes_mallows_score','Normalizedmutualinfoscore','imbalanced_ratio_before_sampling','total_hypersphere','hypersphere_minority','hypersphere_majority','samplesperhypersphere','samplesperhypersphere_minority','samplesperhypersphere_majority','Average distance between class','overlapping','ComplexityMetric','WeightedCM','DualWeightedCM','None','SMOTE','NearMiss','SMOTEENN','Randomoversampling','ADASYN','BorderlineSMOTE','SVMSMOTE','RandomUnderSampler','ClusterCentroids','NearMissversion1','NearMissversion2','NearMissversion3','TomekLinks','EditedNearestNeighbours','RepeatedEditedNearestNeighbours','AllKNN','CondensedNearestNeighbour','NeighbourhoodCleaningRule','InstanceHardnessThreshold','SMOTETomek','KNN','DT','GNB','SVM','RF','F1','G-mean','Accuracy','Precision','Recall','AUC-ROC','AUC-PR','BalancedAccuracy','CWA']
if l == True:
    os.remove(p)
write_file=open(p, 'w',newline='')
csvwriter = csv.writer(write_file)
csvwriter.writerow(fields)
#csvwriter.writerow(["SN", "Name", "Contribution"])          to add each rows
#write_file.close()
#exit(0)
files = sorted(glob.glob("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Dataset/*.csv"))
# spl_word = "\\"
train_time=[]
predict_time=[]
for file_new in files:
    # print(file_new)
    # data = pd.read_csv(file_new)
    # complex_metric = complexity_metric.complexity(data)
    # weighted_complex_metric = wCM.weighted_complexity(data)
    # dualweight_complex_metric = dwCM.dualweighted_complexity(data)
    file_name = file_new
    print("filename:",file_name)
    X, y = data_handling.loading(file_name)
    X= missing_values.handlingMissingValues(X,y,1)
    new_df = pd.DataFrame(data=X[:,:])
    new_df['Class'] = y.tolist()
    data = new_df
    # last_column = data.columns[-1]
    # print(last_column)
    # data.last_column = pd.Categorical(data.last_column)
    # data[last_column] = data.last_column.cat.codes
    # data.Class = data.Class.astype('category').cat.codes
    # le = LabelEncoder()
    # data.Class = le.fit_transform(data.Class)
    # print(data.Class)
    complex_metric = complexity_metric.complexity(data)
    weighted_complex_metric = wCM.weighted_complexity(data)
    dualweight_complex_metric = dwCM.dualweighted_complexity(data)
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
    new_file_name = file_name.replace("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Dataset/", "")

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

    roc_knn=[]
    roc_dt=[]
    roc_SVM=[]
    roc_GNB=[]
    roc_RF=[]
    pr_knn=[]
    pr_dt=[]
    pr_SVM=[]
    pr_GNB=[]
    pr_RF=[]
    bal_knn=[]
    bal_dt=[]
    bal_SVM=[]
    bal_GNB=[]
    bal_RF=[]

    cwa_knn=[]
    cwa_dt=[]
    cwa_SVM=[]
    cwa_GNB=[]
    cwa_RF=[]

    print("KNN classifier")
    print("SVM classifier")
    print("Decision Tree")
    print("")
    for i in ['None','SMOTE','NearMiss','SMOTEENN','Randomoversampling','ADASYN','BorderlineSMOTE','SVMSMOTE','RandomUnderSampler','ClusterCentroids','NearMissversion1','NearMissversion2','NearMissversion3','TomekLinks','EditedNearestNeighbours','RepeatedEditedNearestNeighbours','AllKNN','CondensedNearestNeighbour','NeighbourhoodCleaningRule','InstanceHardnessThreshold','SMOTETomek']:
        None_value = 0
        SMOTE_value = 0
        NearMiss_value = 0
        SMOTEENN_value = 0
        Randomoversampling_value = 0
        ADASYN_value = 0
        BorderlineSMOTE_value = 0
        SVMSMOTE_value = 0
        RandomUnderSampler_value = 0
        ClusterCentroids_value = 0
        NearMissversion1_value = 0
        NearMissversion2_value = 0
        NearMissversion3_value = 0
        TomekLinks_value = 0
        EditedNearestNeighbours_value = 0
        RepeatedEditedNearestNeighbours_value = 0
        AllKNN_value=0
        CondensedNearestNeighbour_value = 0
        OneSidedSelection_value = 0
        NeighbourhoodCleaningRule_value = 0
        InstanceHardnessThreshold_value = 0
        SMOTETomek_value = 0
        if i == 'None':
            uX, uy = X,y
            None_value=1
        elif i == 'SMOTE':
            try:
                # a = np.bincount(y)
                smt = SMOTE()
                uX, uy = smt.fit_sample(X, y)
                SMOTE_value = 1
                # b = np.bincount(uy)
            except:
                continue
        elif i == 'NearMiss':
            try:
                nr = NearMiss()
                uX, uy = nr.fit_sample(X, y)
                NearMiss_value = 1
            except:
                continue
        elif i == 'SMOTEENN':
            try:
                sme = SMOTEENN(random_state=42)
                uX, uy = sme.fit_resample(X, y)
                SMOTEENN_value = 1
            except:
                continue
        elif i=='Randomoversampling':
            try:
                ros = RandomOverSampler(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                Randomoversampling_value = 1
            except:
                continue
        elif i=='ADASYN':
            try:
                ros = ADASYN(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ADASYN_value=1
            except:
                continue
        elif i=='BorderlineSMOTE':
            try:
                ros = BorderlineSMOTE(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                BorderlineSMOTE_value = 1
            except:
                continue
        elif i=='SVMSMOTE':
            try:
                ros = SVMSMOTE(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                SVMSMOTE_value = 1
            except:
                continue
        elif i=='RandomUnderSampler':
            try:
                ros = RandomUnderSampler(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                RandomUnderSampler_value = 1
            except:
                continue
        elif i=='ClusterCentroids':
            try:
                ros = ClusterCentroids(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                ClusterCentroids_value = 1
            except:
                continue
        elif i=='NearMissversion1':
            try:
                ros = NearMiss(version=1)
                uX, uy = ros.fit_resample(X, y)
                NearMissversion1_value = 1
            except:
                continue
        elif i=='NearMissversion2':
            try:
                ros = NearMiss(version=2)
                uX, uy = ros.fit_resample(X, y)
                NearMissversion2_value = 1
            except:
                continue
        elif i=='NearMissversion3':
            try:
                ros = NearMiss(version=3)
                uX, uy = ros.fit_resample(X, y)
                NearMissversion3_value = 1
            except:
                continue
        elif i=='TomekLinks':
            try:
                ros = TomekLinks()
                uX, uy = ros.fit_resample(X, y)
                TomekLinks_value = 1
            except:
                continue
        elif i=='EditedNearestNeighbours':
            try:
                ros = EditedNearestNeighbours()
                uX, uy = ros.fit_resample(X, y)
                EditedNearestNeighbours_value = 1
            except:
                continue
        elif i=='RepeatedEditedNearestNeighbours':
            try:
                ros = RepeatedEditedNearestNeighbours()
                uX, uy = ros.fit_resample(X, y)
                RepeatedEditedNearestNeighbours_value = 1
            except:
                continue
        elif i=='AllKNN':
            try:
                ros = AllKNN()
                uX, uy = ros.fit_resample(X, y)
                AllKNN_value = 1
            except:
                continue
        elif i=='CondensedNearestNeighbour':
            try:
                ros = CondensedNearestNeighbour(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                CondensedNearestNeighbour_value = 1
            except:
                continue
        elif i=='NeighbourhoodCleaningRule':
            try:
                ros = NeighbourhoodCleaningRule()
                uX, uy = ros.fit_resample(X, y)
                NeighbourhoodCleaningRule_value = 1
            except:
                continue
        elif i=='InstanceHardnessThreshold':
            try:
                ros = InstanceHardnessThreshold(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                InstanceHardnessThreshold_value = 1
            except:
                continue
        elif i=='SMOTETomek':
            try:
                ros = SMOTETomek(random_state=42)
                uX, uy = ros.fit_resample(X, y)
                SMOTETomek_value = 1
            except:
                continue
        #no_of_rows_after = uX.shape[0]
        #no_of_columns_after = uX.shape[1]
        #after_sampling=np.bincount(uy)
        #imbalanced_ratio_after_sampling=min(after_sampling)/max(after_sampling)
        for train_index, test_index in kf.split(uX, uy):
            # Splitting out training and testing data
            X_train, X_test = uX[train_index], uX[test_index]
            y_train, y_test = uy[train_index], uy[test_index]
            # knn
            f1_score_knn, accuracy_knn, geometric_mean_knn,precision_knn,recall_knn, auc_roc_knn, auc_pr_knn, bal_acc_knn, cwascore_knn= knn.knn_classifier(X_train, y_train, X_test, y_test, state)
            f1_knn.append(f1_score_knn)
            acc_knn.append(accuracy_knn)
            gmean_knn.append(geometric_mean_knn)
            preci_knn.append(precision_knn.mean())
            rec_knn.append(recall_knn.mean())
            roc_knn.append(auc_roc_knn)
            pr_knn.append(auc_pr_knn)
            bal_knn.append(bal_acc_knn)
            cwa_knn.append(cwascore_knn)
            # decision tree
            f1_score_dt, accuracy_dt, geometric_mean_dt,precision_dt,recall_dt,auc_roc_dt, auc_pr_dt, bal_acc_dt,cwascore_dt = decision_tree.decision_tree(X_train, y_train, X_test, y_test,
                                                                                      state)
            f1_dt.append(f1_score_dt)
            acc_dt.append(accuracy_dt)
            gmean_dt.append(geometric_mean_dt)
            preci_dt.append(precision_dt.mean())
            rec_dt.append(recall_dt.mean())
            roc_dt.append(auc_roc_dt)
            pr_dt.append(auc_pr_dt)
            bal_dt.append(bal_acc_dt)
            cwa_dt.append(cwascore_dt)
            # GaussianNB
            f1_score_GNB, accuracy_GNB, geometric_mean_GNB,precision_GNB,recall_GNB, auc_roc_GNB, auc_pr_GNB, bal_acc_GNB, cwascore_GNB = Gaussian.Gaussian(X_train, y_train, X_test, y_test, state)
            f1_GNB.append(f1_score_GNB)
            acc_GNB.append(accuracy_GNB)
            gmean_GNB.append(geometric_mean_GNB)
            preci_GNB.append(precision_GNB.mean())
            rec_GNB.append(recall_GNB.mean())
            roc_GNB.append(auc_roc_GNB)
            pr_GNB.append(auc_pr_GNB)
            bal_GNB.append(bal_acc_GNB)
            cwa_GNB.append(cwascore_GNB)
            # Support Vector Machine
            # print(i)
            f1_score_SVM, accuracy_SVM, geometric_mean_SVM,precision_SVM,recall_SVM, auc_roc_SVM, auc_pr_SVM, bal_acc_SVM, cwascore_SVM = SupportVM.SVM_classifier(X_train, y_train, X_test, y_test,
                                                                                      state)
            f1_SVM.append(f1_score_SVM)
            acc_SVM.append(accuracy_SVM)
            gmean_SVM.append(geometric_mean_SVM)
            preci_SVM.append(precision_SVM.mean())
            rec_SVM.append(recall_SVM.mean())
            roc_SVM.append(auc_roc_SVM)
            pr_SVM.append(auc_pr_SVM)
            bal_SVM.append(bal_acc_SVM)
            cwa_SVM.append(cwascore_SVM)
            # Random forest
            f1_score_RF, accuracy_RF, geometric_mean_RF,precision_RF,recall_RF, auc_roc_RF, auc_pr_RF, bal_acc_RF, cwascore_RF = random_forest.RF_classifier(X_train, y_train, X_test, y_test,
                                                                                      state)
            f1_RF.append(f1_score_RF)
            acc_RF.append(accuracy_RF)
            gmean_RF.append(geometric_mean_RF)
            preci_RF.append(precision_RF.mean())
            rec_RF.append(recall_RF.mean())
            roc_RF.append(auc_roc_RF)
            pr_RF.append(auc_pr_RF)
            bal_RF.append(bal_acc_RF)
            cwa_RF.append(cwascore_RF)
        # knn
        avg_f1_score_knn = sum(f1_knn) / crossvalidation
        avg_acc_knn = sum(acc_knn) / crossvalidation
        avg_gmean_knn = sum(gmean_knn) / crossvalidation
        avg_precision_knn= sum(preci_knn)/ crossvalidation
        avg_recall_knn=sum(rec_knn)/crossvalidation
        avg_roc_knn=sum(roc_knn)/crossvalidation
        avg_pr_knn=sum(pr_knn)/crossvalidation
        avg_bal_knn=sum(bal_knn)/crossvalidation
        avg_cwa_knn=sum(cwa_knn)/crossvalidation
        # decision tree
        avg_f1_score_dt = sum(f1_dt) / crossvalidation
        avg_acc_dt = sum(acc_dt) / crossvalidation
        avg_gmean_dt = sum(gmean_dt) / crossvalidation
        avg_precision_dt = sum(preci_dt) / crossvalidation
        avg_recall_dt = sum(rec_dt) / crossvalidation
        avg_roc_dt=sum(roc_dt)/crossvalidation
        avg_pr_dt=sum(pr_dt)/crossvalidation
        avg_bal_dt=sum(bal_dt)/crossvalidation
        avg_cwa_dt=sum(cwa_dt)/crossvalidation

        # GaussianNB
        avg_f1_score_GNB = sum(f1_GNB) / crossvalidation
        avg_acc_GNB = sum(acc_GNB) / crossvalidation
        avg_gmean_GNB = sum(gmean_GNB) / crossvalidation
        avg_precision_GNB = sum(preci_GNB) / crossvalidation
        avg_recall_GNB = sum(rec_GNB) / crossvalidation
        avg_roc_GNB=sum(roc_GNB)/crossvalidation
        avg_pr_GNB=sum(pr_GNB)/crossvalidation
        avg_bal_GNB=sum(bal_GNB)/crossvalidation
        avg_cwa_GNB=sum(cwa_GNB)/crossvalidation
        # SVM
        avg_f1_score_SVM = sum(f1_SVM) / crossvalidation
        avg_acc_SVM = sum(acc_SVM) / crossvalidation
        avg_gmean_SVM = sum(gmean_SVM) / crossvalidation
        avg_precision_SVM = sum(preci_SVM) / crossvalidation
        avg_recall_SVM = sum(rec_SVM) / crossvalidation
        avg_roc_SVM=sum(roc_SVM)/crossvalidation
        avg_pr_SVM=sum(pr_SVM)/crossvalidation
        avg_bal_SVM=sum(bal_SVM)/crossvalidation
        avg_cwa_SVM=sum(cwa_SVM)/crossvalidation
        # RF
        avg_f1_score_RF = sum(f1_RF) / crossvalidation
        avg_acc_RF = sum(acc_RF) / crossvalidation
        avg_gmean_RF = sum(gmean_RF) / crossvalidation
        avg_precision_RF = sum(preci_RF) / crossvalidation
        avg_recall_RF = sum(rec_RF) / crossvalidation
        avg_roc_RF=sum(roc_RF)/crossvalidation
        avg_pr_RF=sum(pr_RF)/crossvalidation
        avg_bal_RF=sum(bal_RF)/crossvalidation
        avg_cwa_RF=sum(cwa_RF)/crossvalidation

        csvwriter.writerow([new_file_name, no_of_rows_original,no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over,complex_metric, weighted_complex_metric, dualweight_complex_metric, None_value,SMOTE_value,NearMiss_value,SMOTEENN_value,Randomoversampling_value,ADASYN_value,BorderlineSMOTE_value,SVMSMOTE_value,RandomUnderSampler_value,ClusterCentroids_value,NearMissversion1_value,NearMissversion2_value,NearMissversion3_value,TomekLinks_value,EditedNearestNeighbours_value,RepeatedEditedNearestNeighbours_value,AllKNN_value,CondensedNearestNeighbour_value,NeighbourhoodCleaningRule_value,InstanceHardnessThreshold_value,SMOTETomek_value,1,0,0,0,0,avg_f1_score_knn,avg_gmean_knn,avg_acc_knn,avg_precision_knn,avg_recall_knn,avg_roc_knn,avg_pr_knn,avg_bal_knn,avg_cwa_knn])
        csvwriter.writerow([new_file_name, no_of_rows_original, no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over, complex_metric, weighted_complex_metric, dualweight_complex_metric,None_value,SMOTE_value,NearMiss_value,SMOTEENN_value,Randomoversampling_value,ADASYN_value,BorderlineSMOTE_value,SVMSMOTE_value,RandomUnderSampler_value,ClusterCentroids_value,NearMissversion1_value,NearMissversion2_value,NearMissversion3_value,TomekLinks_value,EditedNearestNeighbours_value,RepeatedEditedNearestNeighbours_value,AllKNN_value,CondensedNearestNeighbour_value,NeighbourhoodCleaningRule_value,InstanceHardnessThreshold_value,SMOTETomek_value,0,1,0,0,0, avg_f1_score_dt,avg_gmean_dt, avg_acc_dt,avg_precision_dt,avg_recall_dt,avg_roc_dt,avg_pr_dt,avg_bal_dt,avg_cwa_dt])
        csvwriter.writerow([new_file_name, no_of_rows_original, no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over, complex_metric, weighted_complex_metric, dualweight_complex_metric,None_value,SMOTE_value,NearMiss_value,SMOTEENN_value,Randomoversampling_value,ADASYN_value,BorderlineSMOTE_value,SVMSMOTE_value,RandomUnderSampler_value,ClusterCentroids_value,NearMissversion1_value,NearMissversion2_value,NearMissversion3_value,TomekLinks_value,EditedNearestNeighbours_value,RepeatedEditedNearestNeighbours_value,AllKNN_value,CondensedNearestNeighbour_value,NeighbourhoodCleaningRule_value,InstanceHardnessThreshold_value,SMOTETomek_value,0,0,1,0,0, avg_f1_score_GNB,avg_gmean_GNB, avg_acc_GNB,avg_precision_GNB,avg_recall_GNB,avg_roc_GNB,avg_pr_GNB,avg_bal_GNB,avg_cwa_GNB])
        csvwriter.writerow([new_file_name, no_of_rows_original, no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over, complex_metric, weighted_complex_metric, dualweight_complex_metric,None_value,SMOTE_value,NearMiss_value,SMOTEENN_value,Randomoversampling_value,ADASYN_value,BorderlineSMOTE_value,SVMSMOTE_value,RandomUnderSampler_value,ClusterCentroids_value,NearMissversion1_value,NearMissversion2_value,NearMissversion3_value,TomekLinks_value,EditedNearestNeighbours_value,RepeatedEditedNearestNeighbours_value,AllKNN_value,CondensedNearestNeighbour_value,NeighbourhoodCleaningRule_value,InstanceHardnessThreshold_value,SMOTETomek_value,0,0,0,1,0,avg_f1_score_SVM,avg_gmean_SVM, avg_acc_SVM,avg_precision_SVM,avg_recall_SVM,avg_roc_SVM,avg_pr_SVM,avg_bal_SVM,avg_cwa_SVM])
        csvwriter.writerow([new_file_name, no_of_rows_original, no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over, complex_metric, weighted_complex_metric, dualweight_complex_metric,None_value,SMOTE_value,NearMiss_value,SMOTEENN_value,Randomoversampling_value,ADASYN_value,BorderlineSMOTE_value,SVMSMOTE_value,RandomUnderSampler_value,ClusterCentroids_value,NearMissversion1_value,NearMissversion2_value,NearMissversion3_value,TomekLinks_value,EditedNearestNeighbours_value,RepeatedEditedNearestNeighbours_value,AllKNN_value,CondensedNearestNeighbour_value,NeighbourhoodCleaningRule_value,InstanceHardnessThreshold_value,SMOTETomek_value, 0,0,0,0,1, avg_f1_score_RF,avg_gmean_RF, avg_acc_RF,avg_precision_RF,avg_recall_RF,avg_roc_RF,avg_pr_RF,avg_bal_RF,avg_cwa_RF])
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

        roc_knn=[]
        roc_dt=[]
        roc_SVM=[]
        roc_GNB=[]
        roc_RF=[]
        pr_knn=[]
        pr_dt=[]
        pr_SVM=[]
        pr_GNB=[]
        pr_RF=[]
        bal_knn=[]
        bal_dt=[]
        bal_SVM=[]
        bal_GNB=[]
        bal_RF=[]

        cwa_knn=[]
        cwa_dt=[]
        cwa_SVM=[]
        cwa_GNB=[]
        cwa_RF=[]
write_file.close()











