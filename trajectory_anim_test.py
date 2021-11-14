import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

r_all = np.load('positions_all_planets.npy')
t_all = np.load('time_planets.npy')

index_0 = 19750
index_1 = 36800
index_2 = 54408
index_tot = index_2
index_list = np.array([index_0, index_1, index_2])

r_planets = np.zeros((2, 3, index_tot))
k = np.array([0, 1, 6])

for i in range(len(k)):
    r_planets[:,i,:] = r_all[:,k[i],:index_tot]

fig, ax = plt.subplots()

# plt.plot(r_planets[0,0,:index_0], r_planets[1,0,:index_0], 'k', lw=0.65)
# plt.plot(r_planets[0,1,:index_1], r_planets[1,1,:index_1], 'k', lw=0.65)
# plt.plot(r_planets[0,2,:index_2], r_planets[1,2,:index_2], 'k', lw=0.65)
plt.plot([0,0], [0,0], 'yo', markersize=10)

line0, = ax.plot(r_planets[0,0,0], r_planets[1,0,0], marker='o')
line1, = ax.plot(r_planets[0,1,0], r_planets[1,1,0], marker='o')
line2, = ax.plot(r_planets[0,2,0], r_planets[1,2,0], marker='o')

def animate(i):
    index = i*40

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    line0.set_data(r_planets[0,0,index], r_planets[1,0,index])
    line1.set_data(r_planets[0,1,index], r_planets[1,1,index])
    line2.set_data(r_planets[0,2,index], r_planets[1,2,index])

    return line0, line1, line2,

ani = FuncAnimation(fig, animate, interval=1, blit=True, frames=1000)

plt.show()
