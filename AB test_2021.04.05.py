import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("D:/USERS/ROMAN/WORK/Python/!проходження курсу CSC 2021/A B test")

df = pd.DataFrame([[1781, 135], [1443, 47]],
                  index=["city", "country"],
                  columns=["for", "against"])

res = sp.stats.chi2_contingency(df, correction=False)
print("p-value: {:.12f}".format(res[1]))
# p-value: 0.000000545387

# #############################################################################

t = 1/100000                   # 0.00001
t                              # 1e-05
"{:.16f}".format(float(t))     # "0.00001"

# A/B tests ###################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2_contingency
import statsmodels.api as sm


os.chdir("D:/USERS/ROMAN/WORK/Python/!проходження курсу CSC 2021/A B test")

df = pd.DataFrame([[1781, 135], [1443, 47]],
                  index=["no seat belt", "seat belt"],
                  columns=["survived", "died"])

#               survived  died
# no seat belt      1781   135
# seat belt         1443    47

s1 = 135                          # число успіхів вибірка А
n1 = 1781 + s1                    # число випробувань вибірка А
s2 = 47                           # число успіхів вибірка Б
n2 = 1443 + s2                    # число випробувань вибірка Б

p1 = s1/n1                        # оцінка імовірності успіху вибірка А
p2 = s2/n2                        # оцінка імовірності успіху вибірка Б
p = (s1 + s2) / (n1 + n2)         # оцінка імовірності успіху вибірка A+Б
z = (p2-p1)/((p*(1-p)*((1/n1)+(1/n2)))**0.5)     # z-мітка

p_value = norm.cdf(z)
# z-мітка і р-значення
print(["{:.12f}".format(a) for a in (abs(z), p_value * 2)])
# ['5.009616324309', '0.000000545387']

# A/B test in python #
z1, p_value1 = sm.stats.proportions_ztest([s1, s2], [n1, n2])
print(["{:.12f}".format(b) for b in (z1, p_value1)])
# ['5.009616324309', '0.000000545387']

# #############################################################################

import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame([[17, 25], [8, 34]],
                  index=["city", "country"],
                  columns=["for", "against"])

res = sp.stats.chi2_contingency(df, correction=True)
print("p-value: {:.12f}".format(res[1]))
# p-value: 0.031732671878   (correction=False) поправка Єйтса ні
# p-value: 0.056246390730   (correction=True)  поправка Єйтса так (більш жорстко)

# #############################################################################






