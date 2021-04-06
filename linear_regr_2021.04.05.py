import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

os.chdir("D:/USERS/ROMAN/WORK/Python/!проходження курсу CSC 2021/linear_regr")

df = pd.read_csv("Albuquerque Home Prices.txt", sep="\t")
df = df.replace(-9999, np.nan)

df.columns          # Index(['PRICE', 'SQFT', 'AGE', 'FEATS', 'NE', 'CUST', 'COR', 'TAX'], dtype='object')
len(df)                                          # 117 rows in df
len(df.dropna(how="any"))                        # 66  rows without NAN

df.apply(lambda x: sum(x.isnull()), axis=0)      # NAN po kolonkah
# PRICE     0
# SQFT      0
# AGE      49
# FEATS     0
# NE        0
# CUST      0
# COR       0
# TAX      10
# dtype: int64

del df["AGE"]

df["TAX"].hist()                                       # rozpodil maije norm tomu zaminiaemo ma seredne
df["TAX"] = df["TAX"].fillna(df["TAX"].mean())         # zamina na seredne
len(df.dropna())
df.corr()

# pobudova modelu
X = df[df.columns.drop(["PRICE"])]
X = df[df.columns.drop(["PRICE", "TAX"])]
X = df[df.columns.drop(["PRICE", "TAX", "NE", "FEATS"])]    # pidsumok
X = df[df.columns.drop(["PRICE", "TAX", "NE", "FEATS", "COR"])]
# X = df[df.columns.drop(["PRICE", "SQFT"])]
y = df["PRICE"]

X = sm.add_constant(X)                                # let's add an intercept (beta_0) to our model
model1 = sm.OLS(y, X).fit()
model1.summary()

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                  PRICE   R-squared:                       0.802
# Model:                            OLS   Adj. R-squared:                  0.791
# Method:                 Least Squares   F-statistic:                     74.37
# Date:                Tue, 06 Apr 2021   Prob (F-statistic):           2.01e-36
# Time:                        23:18:31   Log-Likelihood:                -765.84
# No. Observations:                 117   AIC:                             1546.
# Df Residuals:                     110   BIC:                             1565.
# Df Model:                           6                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         83.1759     63.308      1.314      0.192     -42.286     208.638
# SQFT           0.2920      0.059      4.924      0.000       0.174       0.409
# FEATS         12.1767     12.818      0.950      0.344     -13.225      37.579
# NE             8.0116     35.098      0.228      0.820     -61.544      77.568
# CUST         133.0143     44.756      2.972      0.004      44.319     221.710
# COR          -65.8008     41.839     -1.573      0.119    -148.715      17.113
# TAX            0.5419      0.102      5.303      0.000       0.339       0.744
# ==============================================================================
# Omnibus:                       11.630   Durbin-Watson:                   1.585
# Prob(Omnibus):                  0.003   Jarque-Bera (JB):               33.049
# Skew:                          -0.022   Prob(JB):                     6.66e-08
# Kurtosis:                       5.603   Cond. No.                     8.00e+03
# ==============================================================================

model1.params[0]            # intercept 83.1759
model1.params[1]            # 0.2920
model1.rsquared             # 0.8022
model1.rsquared_adj         # 0.7914











# ##############################################################################################

# os.chdir("D:/USERS/ROMAN/WORK/Python/!проходження курсу CSC 2021")
# base = pd.read_excel("fraud_data_for_import.xlsx", sheet_name = "list1", skiprows = 0)
# # correl = base.corr()
# # base.columns
# # base.dtypes
# # base["sample"].unique()     # array(['train', 'test'], dtype=object)

# base_train = base[base["sample"] == "train"]
# base_test = base[base["sample"] == "test"]

# # full model ===============================================================

# Xtrain = base_train[base_train.columns.drop(["default", "sample"])]
# ytrain = base_train["default"]
# Xtest = base_test[base_test.columns.drop(["default", "sample"])]
# ytest = base_test["default"]

# logit_model = sm.Logit(ytrain, sm.add_constant(Xtrain)).fit()
# logit_model.summary()

# Xtrain.insert(0, "const", 1)
# Xtest.insert(0, "const", 1)
# pred_train = logit_model.predict(Xtrain)
# pred_test = logit_model.predict(Xtest)

# from sklearn.metrics import roc_curve, auc

# # gini na train full -------------------------------------------------------
# actual_train = ytrain.tolist()
# prediction_train = pred_train.tolist()
# fpr, tpr, thresholds = roc_curve(actual_train, prediction_train)
# roc_auc_train = auc(fpr, tpr)
# roc_auc_train
# GINI_train = (2 * roc_auc_train) - 1
# GINI_train                                              # 0.9660914615144907

# # gini na test full --------------------------------------------------------
# actual_test = ytest.tolist()
# prediction_test = pred_test.tolist()
# fpr, tpr, thresholds = roc_curve(actual_test, prediction_test)
# roc_auc_test = auc(fpr, tpr)
# roc_auc_test
# GINI_test = (2 * roc_auc_test) - 1
# GINI_test                                               # 0.9764910105393676

# # pidbir modeli ============================================================

# Xtrain = base_train[["v4", "v8", "v11", "v14", "v28"]]
# Xtrain.corr()
# ytrain = base_train["default"]
# Xtest = base_test[["v4", "v8", "v11", "v14", "v28"]]
# ytest = base_test["default"]

# logit_model = sm.Logit(ytrain, sm.add_constant(Xtrain)).fit()
# logit_model.summary()

# Xtrain.insert(0, "const", 1)
# Xtest.insert(0, "const", 1)
# pred_train = logit_model.predict(Xtrain)
# pred_test = logit_model.predict(Xtest)

# from sklearn.metrics import roc_curve, auc

# # gini na train pidbir -----------------------------------------------------
# actual_train = ytrain.tolist()
# prediction_train = pred_train.tolist()
# fpr, tpr, thresholds = roc_curve(actual_train, prediction_train)
# roc_auc_train = auc(fpr, tpr)
# roc_auc_train
# GINI_train = (2 * roc_auc_train) - 1
# GINI_train                                              # 0.945221204035525

# # gini na test pidbir ------------------------------------------------------
# actual_test = ytest.tolist()
# prediction_test = pred_test.tolist()
# fpr, tpr, thresholds = roc_curve(actual_test, prediction_test)
# roc_auc_test = auc(fpr, tpr)
# roc_auc_test
# GINI_test = (2 * roc_auc_test) - 1
# GINI_test                                               # 0.9682517048977062

# # roc-curve ----------------------------------------------------------------

# plt.figure(figsize = (8, 8))
# plt.title("ROC", fontsize = 17)
# plt.plot(fpr, tpr, "b", label = "AUC = %0.4f"% roc_auc_test)
# plt.legend(loc = "lower right")
# plt.plot([0,1], [0,1], "r--")
# plt.xlim([-0.01, 1.01])
# plt.ylim([-0.01, 1.01])
# plt.ylabel("True Positive Rate (TPR)", fontsize = 17)
# plt.xlabel("False Positive Rate (FPR)", fontsize = 17)
# plt.tick_params(labelsize = 15)
# # plt.show







