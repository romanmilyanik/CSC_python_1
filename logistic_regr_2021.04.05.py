import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

os.chdir("D:/USERS/ROMAN/WORK/Python/!проходження курсу CSC 2021")
base = pd.read_excel("fraud_data_for_import.xlsx", sheet_name = "list1", skiprows = 0)
# correl = base.corr()
# base.columns
# base.dtypes
# base["sample"].unique()     # array(['train', 'test'], dtype=object)

base_train = base[base["sample"] == "train"]
base_test = base[base["sample"] == "test"]

# full model ===============================================================

Xtrain = base_train[base_train.columns.drop(["default", "sample"])]
ytrain = base_train["default"]
Xtest = base_test[base_test.columns.drop(["default", "sample"])]
ytest = base_test["default"]

logit_model = sm.Logit(ytrain, sm.add_constant(Xtrain)).fit()
logit_model.summary()

Xtrain.insert(0, "const", 1)
Xtest.insert(0, "const", 1)
pred_train = logit_model.predict(Xtrain)
pred_test = logit_model.predict(Xtest)

from sklearn.metrics import roc_curve, auc

# gini na train full -------------------------------------------------------
actual_train = ytrain.tolist()
prediction_train = pred_train.tolist()
fpr, tpr, thresholds = roc_curve(actual_train, prediction_train)
roc_auc_train = auc(fpr, tpr)
roc_auc_train
GINI_train = (2 * roc_auc_train) - 1
GINI_train                                              # 0.9660914615144907

# gini na test full --------------------------------------------------------
actual_test = ytest.tolist()
prediction_test = pred_test.tolist()
fpr, tpr, thresholds = roc_curve(actual_test, prediction_test)
roc_auc_test = auc(fpr, tpr)
roc_auc_test
GINI_test = (2 * roc_auc_test) - 1
GINI_test                                               # 0.9764910105393676

# pidbir modeli ============================================================

Xtrain = base_train[["v4", "v8", "v11", "v14", "v28"]]
Xtrain.corr()
ytrain = base_train["default"]
Xtest = base_test[["v4", "v8", "v11", "v14", "v28"]]
ytest = base_test["default"]

logit_model = sm.Logit(ytrain, sm.add_constant(Xtrain)).fit()
logit_model.summary()

Xtrain.insert(0, "const", 1)
Xtest.insert(0, "const", 1)
pred_train = logit_model.predict(Xtrain)
pred_test = logit_model.predict(Xtest)

from sklearn.metrics import roc_curve, auc

# gini na train pidbir -----------------------------------------------------
actual_train = ytrain.tolist()
prediction_train = pred_train.tolist()
fpr, tpr, thresholds = roc_curve(actual_train, prediction_train)
roc_auc_train = auc(fpr, tpr)
roc_auc_train
GINI_train = (2 * roc_auc_train) - 1
GINI_train                                              # 0.945221204035525

# gini na test pidbir ------------------------------------------------------
actual_test = ytest.tolist()
prediction_test = pred_test.tolist()
fpr, tpr, thresholds = roc_curve(actual_test, prediction_test)
roc_auc_test = auc(fpr, tpr)
roc_auc_test
GINI_test = (2 * roc_auc_test) - 1
GINI_test                                               # 0.9682517048977062

# roc-curve ----------------------------------------------------------------

plt.figure(figsize = (8, 8))
plt.title("ROC", fontsize = 17)
plt.plot(fpr, tpr, "b", label = "AUC = %0.4f"% roc_auc_test)
plt.legend(loc = "lower right")
plt.plot([0,1], [0,1], "r--")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel("True Positive Rate (TPR)", fontsize = 17)
plt.xlabel("False Positive Rate (FPR)", fontsize = 17)
plt.tick_params(labelsize = 15)
# plt.show







