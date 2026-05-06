import numpy as np
import math
x1=0.35
x2=0.9
w13=0.1
w23=0.8
w14=0.4
w24=0.6
w35=0.3
w45=0.9
t=0.5
lr=1
def sigmoid(x):
    return 1/(1+np.exp(-x))
def dsigmoid(x):
    return x*(1-x)
for i in range(1,11):
    a3=w13*x1+w23*x2
    y3=sigmoid(a3)
    a4=w14*x1+w24*x2
    y4=sigmoid(a4)
    a5=w35*y3+y4*w45
    y5=sigmoid(a5)
    e5=(t-y5)*dsigmoid(y5)
    e4=dsigmoid(y4)*e5*y4
    e3=dsigmoid(y3)*e5*y4
    w45=w45+lr*e5*y4
    w35=w35+lr*e5*y3
    w13=w13+lr*x1*e3
    w14=w14+lr*x1*e4
    w23=w23+lr*x2*e3
    w24=w24+lr*x2*e4
    print("Step: ",i)
    print("Error =", abs(t - y5))
    print("y3 =", round(y3,4), " y4 =", round(y4,4), " y5 =", round(y5,4))
    print("w13 =", round(w13,4), " w14 =", round(w14,4)," w23 =", round(w23,4), " w24 =", round(w24,4)," w35 =", round(w35,4), " w45 =", round(w45,4))
    print()
