import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spheretocube(theta, phi):
    x = 0.5 * math.sin(theta) * math.cos(phi)
    y = 0.5 * math.sin(theta) * math.sin(phi)
    z = 0.5 * math.cos(theta)
    scale = 2 * max(abs(x), abs(y), abs(z))
    x, y, z = x / scale, y / scale, z / scale
    return x,y,z

X = []
Y = []
Z = []
for phi in list(np.linspace(- math.pi, math.pi, 50)):
    for theta in list(np.linspace(0, math.pi / 2, 50)):
        x, y, z = spheretocube(theta, phi)
        X.append(x)
        Y.append(y)
        Z.append(z)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, Z)
plt.show()