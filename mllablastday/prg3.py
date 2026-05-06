import csv
import numpy as np
with open('mllablastday\playtennis.csv') as f:
    data=list(csv.reader(f))
data=data[1:]
feature=['outlook','temperature','humidity','wind']
def entropy(data):
    labels=[row[-1] for row in data]
    _,c=np.unique(labels,return_counts=True)
    p=c/np.sum(c)
    return -sum(p*np.log2(p))
def info_gain(data,col):
    total=entropy(data)
    values=set([row[col] for row in data])
    weighted=0
    for v in values:
        subset=[row for row in data if row[col]==v]
        weighted+=(len(subset)/len(data))*entropy(subset)
    return total-weighted
def id3(data,feature):
    labels=[row[-1] for row in data]
    if labels.count(labels[0])==len(labels):
        return labels[0]
    gains=[info_gain(data,f) for f in range(len(feature))]
    best=gains.index(max(gains))
    tree={feature[best]:{}}
    values=set([row[best] for row in data])
    for v in values:
        sub=[row[:best]+row[best+1:] for row in data if row[best]==v]
        tree[feature[best]][v]=id3(sub,feature[:best]+feature[best+1:])
    return tree
def predict(tree,sample,feature):
    if isinstance(tree, str): return tree
    key=list(tree.keys())[0]
    return predict(tree[key],sample,feature)
sample=['sunny','cool','normal','strong']
tree=id3(data,feature)
print(tree)
print(predict(tree,sample,feature))