import numpy as np

x = np.array([0.5, -0.3])          # input
w = np.array([[1,2],[3,4]])       # weights

y = np.tanh(np.dot(w,x))          # NN output

J = (1-y**2) * w.sum(axis=1)      # Jacobian

H = np.outer(J,J)                 # Hessian

print("jacobian matrix")
print(J)

print("hessian")
print(H)