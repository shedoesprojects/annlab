import numpy as np
import math

x1=0.35
x2=0.90
lr=1
t=0.5
w13=0.10
w23=0.80
w14=0.40
w24=0.60
w35=0.30
w45=0.90
def sigmoid(x):
    return 1/(1+np.exp(-x))
def dsigmoid(x):
    return x*(1-x)

for i in range(11):
    a3=w13*x1+w23*x2
    y3=sigmoid(a3)
    a4=w14*x1+w24*x2
    y4=sigmoid(a4)
    a5=w35*y3+w45*y4
    y5=sigmoid(a5)

    e5=(t-y5)*dsigmoid(y5)
    e3=dsigmoid(y3)*w35*e5
    e4=dsigmoid(y4)*w45*e5
    w45=w45+lr*e5*y4
    w35=w35+lr*e5*y3
    w13=w13+lr*e3*x1
    w23=w23+lr*e3*x2
    w24=w24+lr*e4*x2
    w14=w14+lr*e4*x1
    print(f"epoch: {i}")
    print(f"error: {round(e5,4)}")
    print(f"w13={round(w13,4)}\nw14={round(w14,4)}\nw23={round(w23,4)}\nw24={round(w24,4)}\nw35={round(w35,4)}\nw45={round(w45,4)}")
    print()





