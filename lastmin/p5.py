import numpy as np
import matplotlib.pyplot as plt
f=lambda x: np.sin(x)+np.cos(x)
x=np.array([0.5,0.5])
h=1e-5

J=[(f(x+h*e)-f(x-h*e))/(2*h) for e in np.eye(2)]
H=[[(f(x+h**(ei+ej))-f(x+h*(ei-ej))-f(x-h*(ei-ej))+f(x-h*(ei+ej)))/(4*h*h) for ej in np.eye(2)] for ei in np.eye(2)]

print("jacobian: ",np.round(J,4))
print("hessian",np.round(H,4))


plt.plot(x,J,label="Jacobian")
plt.plot(x,H,label="Hessian")
plt.legend()
plt.show()