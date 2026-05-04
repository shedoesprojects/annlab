import numpy as np

f=lambda x: np.sin(x[0])+np.cos(x[1])
x=np.array([0.5,0.5])
h=1e-5
J=[(f(x+h*e)-f(x-h*e)/(2*h)) for e in np.eye(2)]
H=[[(f(x+h*(ei+ej))-f(x+h*(ei-ej))-f(x-h*(ei-ej))+f(x+h*(ei+ej))/(4*h*h)) for ej in np.eye(2)] for ei in np.eye(2)]

print("J: ",np.round(J,4))
print("H: ",np.round(H,4))