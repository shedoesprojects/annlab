import math
import numpy as np
import pandas as pd

data=pd.read_csv('mllablastday\DBetes.csv').apply(pd.to_numeric).values
x=data[:,:-1]
y=data[:,-1]
s=int(0.7*len(x))
xt,xte,yt,yte=x[:s],x[s:],y[:s],y[s:]
m={}
for c in np.unique(yt):
    r=xt[yt==c]
    m[c]=[(r[:,i].mean(), r[:,i].std() or 1e-6) for i in range(x.shape[1])]
g=lambda x,mu,s : (1/(math.sqrt(2*math.pi)*s))*np.exp((-(x-mu)**2)/(2*s**2))
def pred(row):
    p={}
    for c in m:
        p[c]=np.prod([g(row[i],*m[c][i]) for i in range(len(row))])
        return max(p,key=p.get)
ypred=[pred(row) for row in xte]
acc=sum(ypred[i]==yte[i] for i in range(len(yte)))/len(yte)*100
print(acc)