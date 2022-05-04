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
reg_kap_l = []
knn_kap_l = []
dpl_kap_l = []
tie_kap_l = []
reg_krp_l = []
knn_krp_l = []
dpl_krp_l = []
tie_krp_l = []
reg_prec_l = []
knn_prec_l = []
dpl_prec_l = []
tie_prec_l = []
#kappa table for all metrics
for i in ['F1', 'G-mean', 'Accuracy', 'Precision', 'Recall','ROC_AUC','PR_AUC','Balanced_Accuracy','CWA']:
    kap_reg = reg_kap[i]
    kap_knn = knn_kap[i]
    kap_dpl = dpl_kap[i]
    new_df = pd.DataFrame()
    new_df['reg'] = kap_reg
    new_df['knn'] = kap_knn
    new_df['dpl'] = kap_dpl
    eq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) == 1]
    neq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) != 1]
    neq_df['max'] = new_df.idxmax(axis=1)
    ties = eq_df.shape[0]
    for idx,name in enumerate(neq_df['max'].value_counts().index.tolist()):
        if name == 'reg':
            reg = neq_df['max'].value_counts()[idx]
        elif name == 'knn':
            knn = neq_df['max'].value_counts()[idx]
        else:
            dpl= neq_df['max'].value_counts()[idx]

    reg_kap_l.append(reg)
    knn_kap_l.append(knn)
    dpl_kap_l.append(dpl)
    tie_kap_l.append(ties)

    kap_reg = reg_krp[i]
    kap_knn = knn_krp[i]
    kap_dpl = dpl_krp[i]
    new_df = pd.DataFrame()
    new_df['reg'] = kap_reg
    new_df['knn'] = kap_knn
    new_df['dpl'] = kap_dpl
    eq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) == 1]
    neq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) != 1]
    neq_df['max'] = new_df.idxmax(axis=1)
    ties = eq_df.shape[0]
    for idx,name in enumerate(neq_df['max'].value_counts().index.tolist()):
        if name == 'reg':
            reg = neq_df['max'].value_counts()[idx]
        elif name == 'knn':
            knn = neq_df['max'].value_counts()[idx]
        else:
            dpl= neq_df['max'].value_counts()[idx]

    reg_krp_l.append(reg)
    knn_krp_l.append(knn)
    dpl_krp_l.append(dpl)
    tie_krp_l.append(ties)

    kap_reg = reg_prec[i]
    kap_knn = knn_prec[i]
    kap_dpl = dpl_prec[i]
    new_df = pd.DataFrame()
    new_df['reg'] = kap_reg
    new_df['knn'] = kap_knn
    new_df['dpl'] = kap_dpl
    eq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) == 1]
    neq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) != 1]
    neq_df['max'] = new_df.idxmax(axis=1)
    ties = eq_df.shape[0]
    for idx,name in enumerate(neq_df['max'].value_counts().index.tolist()):
        if name == 'reg':
            reg = neq_df['max'].value_counts()[idx]
        elif name == 'knn':
            knn = neq_df['max'].value_counts()[idx]
        else:
            dpl= neq_df['max'].value_counts()[idx]

    reg_prec_l.append(reg)
    knn_prec_l.append(knn)
    dpl_prec_l.append(dpl)
    tie_prec_l.append(ties)
# f1_kap_reg = reg_kap['F1']
# f1_kap_knn = knn_kap['F1']
# f1_kap_dpl = dpl_kap['F1']

# new_df = pd.DataFrame()
# new_df['f1_reg'] = f1_kap_reg
# new_df['f1_knn'] = f1_kap_knn
# new_df['f1_dpl'] = f1_kap_dpl

# eq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) == 1]
# print(eq_df)
# neq_df = new_df[new_df.apply(pd.Series.nunique, axis=1) != 1]
# neq_df['max'] = new_df.idxmax(axis=1)
# ties = eq_df.shape[0]
# item_counts = neq_df['max'].value_counts()
# print(neq_df.head(5))
# print(ties)
# for idx,name in enumerate(neq_df['max'].value_counts().index.tolist()):
#     if name == 'f1_reg':
#         reg_f1 = neq_df['max'].value_counts()[idx]
#     elif name == 'f1_knn':
#         knn_f1 = neq_df['max'].value_counts()[idx]
#     else:
#         dpl_f1 = neq_df['max'].value_counts()[idx]

# print(reg_f1, knn_f1, dpl_f1)

df = pd.DataFrame({'Metrics':['F1', 'G-mean', 'Accuracy', 'Precision', 'Recall','ROC_AUC','PR_AUC','Balanced_Accuracy','CWA']})
df['k_RG'] = reg_kap_l
df['k_NN'] = knn_kap_l
df['k_DPL'] = dpl_kap_l
df['k_Tie'] = tie_kap_l

df['kr_RG'] = reg_krp_l
df['kr_NN'] = knn_krp_l
df['kr_DPL'] = dpl_krp_l
df['kr_Tie'] = tie_krp_l

df['p_RG'] = reg_prec_l
df['p_NN'] = knn_prec_l
df['p_DPL'] = dpl_prec_l
df['p_Tie'] = tie_prec_l
df.to_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Tables/table3.csv", index=False)
