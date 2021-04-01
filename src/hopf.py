import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

num_osc = 1
dt = 0.001

def hopf(x, y, omega):
    r2 = x**2 + y**2
    rng = omega * np.arange(1, num_osc + 1)
    x = x + ((1 - r2) * x - rng * y) * dt
    y = y + ((1 - r2) * y + rng * x) * dt
    return x, y


r = np.ones((num_osc,))
phi = np.zeros((num_osc,))
z = r * np.exp(1j * phi)
x = np.real(z)
y = np.imag(z)
omega = 30
X = []
Y = []
fig, ax = plt.subplots(2, 1, figsize = (5,10))
for i in tqdm(range(1000)):
    X.append(x)
    Y.append(y)
    x, y = hopf(x, y, omega)
#X = np.concatenate(X, 0)
#Y = np.concatenate(Y, 0)
print('Plotting')
X = [x[0] for x in X]
Y = [y[0] for y in Y]
#print(X)
#print(Y)
ax[0].plot(X)
ax[1].plot(Y)
fig.savefig('hopf.png')
plt.show()

