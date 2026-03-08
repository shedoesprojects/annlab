import sympy as sp

x,y = sp.symbols('x y')

f = x**2 + y**2 + x*y

# Jacobian
J = [sp.diff(f,x), sp.diff(f,y)]

# Hessian
H = sp.hessian(f,(x,y))

print("Jacobian Matrix:")
print(J)

print("\nHessian Matrix:")
print(H)