import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data=pd.read_csv('mllablastday\kmeans.csv')
model=KMeans(n_clusters=2)
model.fit(data)
print(model.labels_)
print(model.cluster_centers_)