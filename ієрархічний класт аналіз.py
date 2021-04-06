import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as plt
import seaborn as sns

os.chdir("D:/USERS/ROMAN/WORK/Python/!проходження курсу CSC 2021/ієрархічний кластерний аналіз")

df = pd.read_csv("beverage_r.csv", sep = ";", index_col = "numb.obs")

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

link = linkage(df, "ward", "euclidean")
dn = dendrogram(link, orientation = "top")
df["cluster"] = fcluster(link, 3, criterion = "distance")
df.groupby("cluster").size()


















