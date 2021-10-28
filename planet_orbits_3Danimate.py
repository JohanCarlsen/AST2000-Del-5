import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


r_all = np.load('positions_all_planets.npy')
time = np.load('time_planets.npy')
print(np.shape(r_all))

fig = plt.figure(facecolor='k')
ax = fig.add_subplot(projection='3d', facecolor='k')

for i in range(8):
    ax.plot(r_all[0,i,:], r_all[1,i,:], 0)

ax.plot(0, 0, 0, color='yellow', marker='o', markersize=10)

def update(frame):
    p0, = ax.plot3D(r_all[0,0,frame], r_all[1,0,frame], 0, color='blue', marker='o', markersize=5)
    p1, = ax.plot3D(r_all[0,1,frame], r_all[1,1,frame], 0, color='orange', marker='o', markersize=5)
    p2, = ax.plot3D(r_all[0,2,frame], r_all[1,2,frame], 0, color='green', marker='o', markersize=5)
    p3, = ax.plot3D(r_all[0,3,frame], r_all[1,3,frame], 0, color='darkred', marker='o', markersize=5)
    p4, = ax.plot3D(r_all[0,4,frame], r_all[1,4,frame], 0, color='purple', marker='o', markersize=5)
    p5, = ax.plot3D(r_all[0,5,frame], r_all[1,5,frame], 0, color='brown', marker='o', markersize=5)
    p6, = ax.plot3D(r_all[0,6,frame], r_all[1,6,frame], 0, color='pink', marker='o', markersize=5)
    p7, = ax.plot3D(r_all[0,7,frame], r_all[1,7,frame], 0, color='gray', marker='o', markersize=5)
    timelabel = ax.text(10, 5, 0.05, f't={time[frame]:.1f}yr', fontsize=16)
    timelabel.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))
    return p0, p1, p2, p3, p4, p5, p6, p7, timelabel


# Setting the axes properties
ax.set_xlim3d([-20, 20])
ax.set_xlabel('X')

ax.set_ylim3d([-20, 20])
ax.set_ylabel('Y')

# ax.set_zlim3d([-0.1, 0.1])
ax.set_zlabel('Z')
ax.axis('off')

ani = animation.FuncAnimation(fig, func=update, frames=range(0, len(r_all[0,0,:]), 500), interval=10, blit=True)

plt.show()
