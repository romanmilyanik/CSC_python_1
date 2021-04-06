import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("D:/USERS/ROMAN/WORK/Python/!проходження курсу CSC 2021/k-mens")

df = pd.read_csv("beverage.csv", sep = ";", index_col = "numb.obs")

from sklearn.cluster import KMeans

model = KMeans(n_clusters = 2,  random_state = 42)
model.fit(df)
model.labels_
model.cluster_centers_

new_items = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

model.predict(new_items)

# vuznachutu chuslo klasteriv

K = range(1, 11)
models = [KMeans(n_clusters = k,  random_state = 42).fit(df) for k in K]
dist = [model.inertia_ for model in models]

plt.plot(K, dist, marker='o')
plt.xlabel("k")
plt.ylabel("Sum of distances")
plt.title("The Elbow Method showing the optimal k")
plt.show()

# vugladae scho e 3 klastera

model = KMeans(n_clusters = 3,  random_state = 42)
model.fit(df)
df["cluster"] = model.labels_
df.groupby("cluster").mean()
df.groupby("cluster").size()

### example 2 ====================================================================
import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("D:/USERS/ROMAN/WORK/Python/!проходження курсу CSC 2021/k-mens")

df = pd.read_csv("assess.dat", sep = "\t", index_col = "NAME")
from sklearn.cluster import KMeans

del df["NR"]

model = KMeans(n_clusters=4, random_state=42)
model.fit(df)
df["cluster"] = model.labels_
df.groupby("cluster").mean()

# vuznachutu chuslo klasteriv

K = range(1, 11)
models = [KMeans(n_clusters = k,  random_state = 42).fit(df) for k in K]
dist = [model.inertia_ for model in models]

plt.plot(K, dist, marker='o')
plt.xlabel("k")
plt.ylabel("Sum of distances")
plt.title("The Elbow Method showing the optimal k")
plt.show()

# vugladae scho e 4 klastera

### example 3 ====================================================================
import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("D:/USERS/ROMAN/WORK/Python/!проходження курсу CSC 2021/k-mens")

df = pd.read_csv("Protein Consumption in Europe.csv", sep = ";", decimal = ",", index_col = "Country")
from sklearn.cluster import KMeans
from sklearn import preprocessing

norm = preprocessing.StandardScaler()
norm.fit(df)
X = norm.transform(df)

K = range(1, 16)
models = [KMeans(n_clusters=k).fit(X) for k in K]
dist = [model.inertia_ for model in models]

plt.plot(K, dist, marker="o")
plt.xlabel("k")
plt.ylabel("Sum of distances")
plt.title("The Elbow Method showing the optimal k")
plt.show()


model = KMeans(n_clusters=6)
model.fit(X)
df["cluster"] = model.labels_
df.groupby("cluster").mean()
df["cluster"].sort_values()





















