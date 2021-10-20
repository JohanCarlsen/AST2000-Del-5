import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 500)
r = 3

x = r*np.cos(theta)
y = r*np.sin(theta)

theta_p = np.linspace(0, 2*np.pi, 20)

xr = np.linspace(0, 3, 40)
f = -np.sin(xr) - 0.5*xr
i = 0
for tt in theta_p:
    x_p = r*np.cos(tt)
    y_p = r*np.sin(tt)
    for j in range(0, 5):
        if i == len(xr):
            break
        plt.plot(x,y ,'k', linewidth=2)
        plt.axis('off')
        plt.plot(f[i], xr[i], 'bo', label='Rocket')
        plt.plot(x_p, y_p, color='r', marker='o', markersize=16, label='Planet')
        plt.legend()
        plt.draw()
        plt.pause(0.1)
        plt.clf()
        i += 1
