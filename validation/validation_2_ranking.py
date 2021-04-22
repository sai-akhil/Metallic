import pandas as pd
import numpy as np
import xgboost as xg
from sklearn import metrics
import os
from os import path
import csv
import sklearn
import krippendorff
p = 'Metafeatures/kappa_score.csv'
p1='Metafeatures/krippendorf.csv'
p2='Metafeatures/precission.csv'
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
df = pd.read_csv('Metafeatures/features.csv')
dataset_names=list(df['Dataset'])
names=np.unique(dataset_names)
kapa_score=[]
krippen_score=[]
precisson_5=[]
for j in ['KNN','DT','GNB','SVM','RF']:
    for i in names:
        kapa_score=[]
        krippen_score=[]
        precisson_5=[]
        for metric in ['F1','G-mean','Accuracy','Precision','Recall']:
            dataframe = df[df['Dataset'] != i]
            x_test_sample = df[df['Dataset'] == i]
            x_test_selected = x_test_sample[x_test_sample[j] == 1]
            y_test = np.array(x_test_selected[metric])
            x_test = x_test_selected.iloc[:, 1:46]
            x_test_sample=x_test.iloc[:,24:]
            count1=1
            sample_not_present=[]
            for kk in range(0,21):
                sample1=list(x_test_sample.iloc[:,kk])
                if 1 not in sample1:
                    sample_not_present.append(count1)
                count1=count1+1
            selected_dataframe = dataframe[dataframe[j] == 1]
            y_train = np.array(selected_dataframe[metric])
            x_train = selected_dataframe.iloc[:, 1:46]
            model = xg.XGBRegressor(objective ='reg:squarederror',colsample_bytree=0.4, gamma=0, learning_rate=0.07, max_depth=3,
                                    min_child_weight=1.5,n_estimators=10000, reg_alpha=0.75, reg_lambda=0.45, subsample=0.6, seed=42)
            x_train = np.array(x_train)
            model.fit(np.array(x_train), y_train)
            y_pred = list(model.predict(np.array(x_test)))
            if len(y_pred) != 21:
                for ii in sample_not_present:
                    y_pred.insert(int(ii)-1,0)
            y_pred = pd.DataFrame(y_pred)
            ranking = y_pred.rank(ascending=False,method="first")
            ranking = np.array(ranking)
            pred_ranking=[]
            for ij in ranking:
                for jj in ij:
                    pred_ranking.append(jj)
            y_test=list(y_test)
            if len(y_test)!=21:
                for ij in sample_not_present:
                    y_test.insert(int(ij)-1,0)
            y_test=pd.DataFrame(y_test)

            ranking1 = y_test.rank(ascending=False,method="first")
            ranking1 = np.array(ranking1)
            real_ranking = []
            for ij in ranking1:
                for jj in ij:
                    real_ranking.append(jj)
            print(real_ranking)
            print(pred_ranking)
            kapa_score.append(sklearn.metrics.cohen_kappa_score(np.array(real_ranking),np.array(pred_ranking)))

            reliability_data=[list(real_ranking),list(pred_ranking)]
            krippen_score.append(krippendorff.alpha(reliability_data=reliability_data))
            real_5=[]
            pred_5=[]
            for r in range(1,6):
                if r in real_ranking:
                    index=real_ranking.index(r)
                    real_5.append(index+1)
                if r in pred_ranking:
                    index1=pred_ranking.index(r)
                    pred_5.append(index1+1)
            count=0
            for rk in pred_5:
                if rk in real_5:
                    count=count+1
            precisson_5.append(count/5)
            data = {'Dataset':[i]*21,'Classifier':[j]*21,'Metric':[metric]*21,'predicted_score': list(y_pred[0]),'predicted_ranking': list(pred_ranking),'original_score': list(y_test[0]),'original_ranking':list(real_ranking)}
            new_file=pd.DataFrame(data)
            path='test_file_result/'+j+'_'+metric+'_'+i
            new_file.to_csv(path,index=False)
            data={}


        csvwriter.writerow([i, j, kapa_score[0], kapa_score[1], kapa_score[2], kapa_score[3], kapa_score[4]])
        csvwriter1.writerow([i, j, krippen_score[0], krippen_score[1], krippen_score[2], krippen_score[3], krippen_score[4]])
        csvwriter2.writerow([i, j, precisson_5[0], precisson_5[1], precisson_5[2], precisson_5[3], precisson_5[4]])
write_file.close()
write_file1.close()
write_file2.close()
