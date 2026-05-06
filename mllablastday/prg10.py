#10. Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select the appropriate data set for your experiment and draw graphs.

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,10,20)
y=np.sin(x)
tau=0.5
ypred=[]
for i in x:
    w=np.exp((-(x-i)**2)/(2*tau**2))
    W=np.diag(w)
    x_mat=np.vstack([np.ones(len(x)),x]).T
    theta=np.linalg.inv(x_mat.T@W@x_mat)@x_mat.T@W@y
    ypred.append(theta[0]+theta[1]*i)
plt.scatter(x,y,label="Data")
plt.plot(x,ypred,color='red',label="LWR")
plt.legend()
plt.show()