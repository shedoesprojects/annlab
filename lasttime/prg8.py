from sklearn.cluster import KMeans
import pandas as pd

data=pd.read_csv('mllablastday\kmeans.csv')

model=KMeans(n_clusters=3)
model.fit(data)

print(model.labels_)
print(model.cluster_centers_)