import numpy as np

vx = -0.93240737
vy = -0.86473146
v = np.sqrt(vx**2 + vy**2)
ex = 0.01
ey = 0.01
vx_hat = vx+ex
vy_hat = vy+ey
v_hat = np.sqrt(vx_hat**2 + vy_hat**2)
G = 1
m = 1
x = 0.97000436
y = 0.24308753
R = np.sqrt(x**2 + y**2)
B = (5/2) * G * m**2
C = (3/4) * m
R_hat = B/((B/R) + C*(2*ex*vx + 2*ey*vy + ex**2 + ey**2))
KE = (3/4) * m * v**2
PE = -(5/2) * (G * m**2) / R
KE_hat = (3/4) * m * v_hat**2
PE_hat = -(5/2) * (G * m**2) / R_hat
print(R)
print(KE+PE)
print(R_hat)
print(KE_hat+PE_hat)
print(B/R_hat - B/R)
print(C*v_hat**2 - C*v**2)