import pandas as pd
import warnings
warnings.filterwarnings("ignore")
reg_kap = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/regression_scores/kappa_score.csv")
reg_krp = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/regression_scores/krippendorf.csv")
reg_prec = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/regression_scores/precission.csv")

knn_kap = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/knn_scores/kappa_score_knn.csv")
knn_krp = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/knn_scores/krippendorf.csv")
knn_prec = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/knn_scores/precission.csv")

dpl_kap = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/deep_scores/kappa_score.csv")
dpl_krp = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/deep_scores/krippendorf.csv")
dpl_prec = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/deep_scores/precission.csv")

mse_reg = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/sample/mean_square_error.csv")
mse_dpl = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/sample/deep_mean_square_error.csv")


# dpl_kap = dpl_kap.loc[reg_kap['Classifier'] == 'SVM']
# dpl_krp = dpl_krp.loc[reg_krp['Classifier'] == 'SVM']
# dpl_prec = dpl_prec.loc[reg_prec['Classifier'] == 'SVM']
# mse_dpl = mse_dpl.loc[mse_dpl['Classifier'] == 'SVM']

    #for regression
for i in ['F1','G-mean', 'Accuracy', 'Precision', 'Recall','ROC_AUC','PR_AUC','Balanced_Accuracy','CWA']:
    print("For :",i)
    # print("")
    # print("Kappa scores")
    # # print("Regression")
    # # kap_reg = reg_kap[i]
    # # reg_kap_mean = kap_reg.mean()
    # # reg_kap_mean = str(round(reg_kap_mean, 3))
    # # print("Mean:",reg_kap_mean)
    # # reg_kap_std = kap_reg.std()
    # # reg_kap_std = str(round(reg_kap_std, 3))
    # # print("Std Dev:",reg_kap_std)
    # # print("")
    # # print("KNN")
    # # kap_knn = knn_kap[i]
    # # reg_kap_mean = kap_knn.mean()
    # # reg_kap_mean = str(round(reg_kap_mean, 3))
    # # print("Mean:",reg_kap_mean)
    # # reg_kap_std = kap_knn.std()
    # # reg_kap_std = str(round(reg_kap_std, 3))
    # # print("Std Dev:",reg_kap_std)
    # print("")
    # print("DNN")
    # kap_dpl = dpl_kap[i]
    # reg_kap_mean = kap_dpl.mean()
    # reg_kap_mean = str(round(reg_kap_mean, 3))
    # print("Mean:",reg_kap_mean)
    # reg_kap_std = kap_dpl.std()
    # reg_kap_std = str(round(reg_kap_std, 3))
    # print("Std Dev:",reg_kap_std)
    # print("")
    # print("Kripp scores")
    # # print("Regression")
    # # kap_reg = reg_krp[i]
    # # reg_kap_mean = kap_reg.mean()
    # # reg_kap_mean = str(round(reg_kap_mean, 3))
    # # print("Mean:",reg_kap_mean)
    # # reg_kap_std = kap_reg.std()
    # # reg_kap_std = str(round(reg_kap_std, 3))
    # # print("Std Dev:",reg_kap_std)
    # # print("")
    # # print("KNN")
    # # kap_knn = knn_krp[i]
    # # reg_kap_mean = kap_knn.mean()
    # # reg_kap_mean = str(round(reg_kap_mean, 3))
    # # print("Mean:",reg_kap_mean)
    # # reg_kap_std = kap_knn.std()
    # # reg_kap_std = str(round(reg_kap_std, 3))
    # # print("Std Dev:",reg_kap_std)
    # print("")
    # print("DNN")
    # kap_dpl = dpl_krp[i]
    # reg_kap_mean = kap_dpl.mean()
    # reg_kap_mean = str(round(reg_kap_mean, 3))
    # print("Mean:",reg_kap_mean)
    # reg_kap_std = kap_dpl.std()
    # reg_kap_std = str(round(reg_kap_std, 3))
    # print("Std Dev:",reg_kap_std)
    # print("")

    # print("Precisiion scores")
    # # print("Regression")
    # # kap_reg = reg_prec[i]
    # # reg_kap_mean = kap_reg.mean()
    # # reg_kap_mean = str(round(reg_kap_mean, 3))
    # # print("Mean:",reg_kap_mean)
    # # reg_kap_std = kap_reg.std()
    # # reg_kap_std = str(round(reg_kap_std, 3))
    # # print("Std Dev:",reg_kap_std)
    # # print("")
    # # print("KNN")
    # # kap_knn = knn_prec[i]
    # # reg_kap_mean = kap_knn.mean()
    # # reg_kap_mean = str(round(reg_kap_mean, 3))
    # # print("Mean:",reg_kap_mean)
    # # reg_kap_std = kap_knn.std()
    # # reg_kap_std = str(round(reg_kap_std, 3))
    # # print("Std Dev:",reg_kap_std)
    # print("")
    # print("DNN")
    # kap_dpl = dpl_prec[i]
    # reg_kap_mean = kap_dpl.mean()
    # reg_kap_mean = str(round(reg_kap_mean, 3))
    # print("Mean:",reg_kap_mean)
    # reg_kap_std = kap_dpl.std()
    # reg_kap_std = str(round(reg_kap_std, 3))
    # print("Std Dev:",reg_kap_std)
    # print("")

    print("MSE Errors")
    print("Regression")
    kap_reg = mse_reg[i]
    reg_kap_mean = kap_reg.mean()
    reg_kap_mean = str(round(reg_kap_mean, 3))
    print("Mean:",reg_kap_mean)
    reg_kap_std = kap_reg.std()
    reg_kap_std = str(round(reg_kap_std, 3))
    print("Std Dev:",reg_kap_std)
    print("")
    print("DPL")
    kap_knn = mse_dpl[i]
    reg_kap_mean = kap_knn.mean()
    reg_kap_mean = str(round(reg_kap_mean, 3))
    print("Mean:",reg_kap_mean)
    reg_kap_std = kap_knn.std()
    reg_kap_std = str(round(reg_kap_std, 3))
    print("Std Dev:",reg_kap_std)
    print("")
    # reg_kap_mean = kap_reg.mean()
    # reg_kap_mean = str(round(reg_kap_mean, 3))
    # reg_f1_avg.append(reg_kap_mean)
    # reg_kap_std = kap_reg.std()
    # reg_kap_std = str(round(reg_kap_std, 3))
    # reg_f1_sd.append(reg_kap_std)


    # kap_knn = knn_kap[i]
    # kap_dpl = dpl_kap[i]
#     new_df = pd.DataFrame()
#     new_df['reg'] = kap_reg
#     new_df['knn'] = kap_knn
#     new_df['dpl'] = kap_dpl
#     eq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) == 1]
#     neq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) != 1]
#     neq_df['max'] = new_df.idxmax(axis=1)
#     ties = eq_df.shape[0]
#     for idx,name in enumerate(neq_df['max'].value_counts().index.tolist()):
#         if name == 'reg':
#             reg = neq_df['max'].value_counts()[idx]
#         elif name == 'knn':
#             knn = neq_df['max'].value_counts()[idx]
#         else:
#             dpl= neq_df['max'].value_counts()[idx]

#     reg_kap_l.append(reg)
#     knn_kap_l.append(knn)
#     dpl_kap_l.append(dpl)
#     tie_kap_l.append(ties)

#     kap_reg = reg_krp[i]
#     kap_knn = knn_krp[i]
#     kap_dpl = dpl_krp[i]
#     new_df = pd.DataFrame()
#     new_df['reg'] = kap_reg
#     new_df['knn'] = kap_knn
#     new_df['dpl'] = kap_dpl
#     eq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) == 1]
#     neq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) != 1]
#     neq_df['max'] = new_df.idxmax(axis=1)
#     ties = eq_df.shape[0]
#     for idx,name in enumerate(neq_df['max'].value_counts().index.tolist()):
#         if name == 'reg':
#             reg = neq_df['max'].value_counts()[idx]
#         elif name == 'knn':
#             knn = neq_df['max'].value_counts()[idx]
#         else:
#             dpl= neq_df['max'].value_counts()[idx]

#     reg_krp_l.append(reg)
#     knn_krp_l.append(knn)
#     dpl_krp_l.append(dpl)
#     tie_krp_l.append(ties)

#     kap_reg = reg_prec[i]
#     kap_knn = knn_prec[i]
#     kap_dpl = dpl_prec[i]
#     new_df = pd.DataFrame()
#     new_df['reg'] = kap_reg
#     new_df['knn'] = kap_knn
#     new_df['dpl'] = kap_dpl
#     eq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) == 1]
#     neq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) != 1]
#     neq_df['max'] = new_df.idxmax(axis=1)
#     ties = eq_df.shape[0]
#     for idx,name in enumerate(neq_df['max'].value_counts().index.tolist()):
#         if name == 'reg':
#             reg = neq_df['max'].value_counts()[idx]
#         elif name == 'knn':
#             knn = neq_df['max'].value_counts()[idx]
#         else:
#             dpl= neq_df['max'].value_counts()[idx]

#     reg_prec_l.append(reg)
#     knn_prec_l.append(knn)
#     dpl_prec_l.append(dpl)
#     tie_prec_l.append(ties)

# df = pd.DataFrame({'Metrics':['F1', 'G-mean', 'Accuracy', 'Precision', 'Recall','ROC_AUC','PR_AUC','Balanced_Accuracy','CWA']})
# df['k_RG'] = reg_kap_l
# df['k_NN'] = knn_kap_l
# df['k_DPL'] = dpl_kap_l
# df['k_Tie'] = tie_kap_l

# df['kr_RG'] = reg_krp_l
# df['kr_NN'] = knn_krp_l
# df['kr_DPL'] = dpl_krp_l
# df['kr_Tie'] = tie_krp_l

# df['p_RG'] = reg_prec_l
# df['p_NN'] = knn_prec_l
# df['p_DPL'] = dpl_prec_l
# df['p_Tie'] = tie_prec_l
# df.to_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Tables/table3.csv", index=False)
