from gpaw.test import findpeak
import numpy as np
from scipy.ndimage import gaussian_filter1d


# collect data from df_*.csv files to check omega peak equality
df_tetra = np.loadtxt('df_tetra.csv', delimiter=',')
df_point = np.loadtxt('df_point.csv', delimiter=',')
df_wimag = gaussian_filter1d(df_point[:, 4], 7)
w_w = df_tetra[:, 0][200:450]

# find the peaks for each intensity profile
w1, I1 = findpeak(w_w, df_tetra[:, 4][200:450])
w2, I2 = findpeak(w_w, df_point[:, 4][200:450])
w3, I3 = findpeak(w_w, df_wimag[200:450])

# check that the omega peak for tetra and point integration methods agree
# and check the point integration peak is aprx the same as the tetra peak
assert abs(w1 - w2) < 0.015
assert abs(w1 - w3) < 0.05
