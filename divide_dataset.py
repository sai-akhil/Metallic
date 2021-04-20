import numpy as np
import glob
import data_handling,os,csv
from os import path
p ='Metafeatures/extra.csv'
l = path.exists(p)
fields=['Dataset', 'Original Rows', 'Columns','no of classes','Minority class ','imbalanced_ratio']
if l == True:
    os.remove(p)
write_file=open(p, 'w',newline='')
csvwriter = csv.writer(write_file)
csvwriter.writerow(fields)
files = glob.glob("C:/Users/DELL/PycharmProjects/metalearning_version1/side_dataset/*.csv")
spl_word = "\\"
rows=[]
imbalance=[]
columns=[]
muti=0
bin=0
count=1
print("file name\t\t\trows \t\t\t column \t\t\t nr classes\t\t\t Minority class \t\t\t imbalanced_ratio")
for file in files:
    file_name = file.partition(spl_word)[2]

    X, y = data_handling.loading(file_name)
    no_of_rows_original = X.shape[0]
    no_of_columns_original = X.shape[1]
    no_of_class=len(np.unique(y))
    if no_of_class > 2:
        state='multiclass'
        muti=muti+1
        state_value=1
    else:
        state='binaryclass'
        bin=bin+1
    y = y.astype(int)
    classes_data = list(np.unique(y))
    list_of_instance = [sum(y == c) for c in classes_data]
    min_instance = min(list_of_instance)
    max_instance = max(list_of_instance)
    imbalanced_ratio_before_sampling = min_instance / max_instance
    csvwriter.writerow([file_name, no_of_rows_original, no_of_columns_original,no_of_class,min_instance,imbalanced_ratio_before_sampling])

    print(count,file_name,"\t\t\t",no_of_rows_original,"\t\t\t",no_of_columns_original,"\t\t\t",no_of_class,"\t\t\t",min_instance,"\t\t\t",imbalanced_ratio_before_sampling)

    count=count+1
    rows.append(no_of_rows_original)
    columns.append(no_of_columns_original)
    imbalance.append(imbalanced_ratio_before_sampling)
print(min(imbalance))
print(max(imbalance))
print(bin)
print(muti)