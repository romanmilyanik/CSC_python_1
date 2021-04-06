import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as plt
import seaborn as sns
import statsmodels.api as sm



os.chdir("D:/USERS/ROMAN/WORK/Python/!проходження курсу CSC 2021")
base = pd.read_csv("fraud_data.csv", sep = ",")

writer = pd.ExcelWriter("towrite.xlsx", engine="xlsxwriter")
base.to_excel(writer, index = False)
writer.save()


















