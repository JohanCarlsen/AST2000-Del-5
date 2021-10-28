import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig = plt.figure()
ax = fig.add_subplot()
ax.axis('off')

r_all = np.load('positions_all_planets.npy')
time = np.load('time_planets.npy')

plt.subplots_adjust(bottom=0.25)

N = 1000

l0, = plt.plot(r_all[0,0,::N], r_all[1,0,::N], lw=2)
l1, = plt.plot(r_all[0,1,::N], r_all[1,1,::N], lw=2)
l2, = plt.plot(r_all[0,2,::N], r_all[1,2,::N], lw=2)
l3, = plt.plot(r_all[0,3,::N], r_all[1,3,::N], lw=2)
l4, = plt.plot(r_all[0,4,::N], r_all[1,4,::N], lw=2)
l5, = plt.plot(r_all[0,5,::N], r_all[1,5,::N], lw=2)
l6, = plt.plot(r_all[0,6,::N], r_all[1,6,::N], lw=2)
l7, = plt.plot(r_all[0,7,::N], r_all[1,7,::N], lw=2)

p0, = ax.plot(r_all[0,0,0], r_all[1,0,0], color='blue', marker='o', markersize=6)
p1, = ax.plot(r_all[0,1,0], r_all[1,1,0], color='orange', marker='o', markersize=6)
p2, = ax.plot(r_all[0,2,0], r_all[1,2,0], color='green', marker='o', markersize=6)
p3, = ax.plot(r_all[0,3,0], r_all[1,3,0], color='darkred', marker='o', markersize=6)
p4, = ax.plot(r_all[0,4,0], r_all[1,4,0], color='purple', marker='o', markersize=6)
p5, = ax.plot(r_all[0,5,0], r_all[1,5,0], color='brown', marker='o', markersize=6)
p6, = ax.plot(r_all[0,6,0], r_all[1,6,0], color='pink', marker='o', markersize=6)
p7, = ax.plot(r_all[0,7,0], r_all[1,7,0], color='gray', marker='o', markersize=6)

# ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

stime = Slider(axtime, 'Time [yr]', 0, 40)

def update(val):
    T = stime.val
    dt = time[1] - time[0]
    index = int(T / dt)
    # print(time)
    p0.set_data(r_all[0, 0, index], r_all[1, 0, index])
    p1.set_data(r_all[0, 1, index], r_all[1, 1, index])
    p2.set_data(r_all[0, 2, index], r_all[1, 2, index])
    p3.set_data(r_all[0, 3, index], r_all[1, 3, index])
    p4.set_data(r_all[0, 4, index], r_all[1, 4, index])
    p5.set_data(r_all[0, 5, index], r_all[1, 5, index])
    p6.set_data(r_all[0, 6, index], r_all[1, 6, index])
    p7.set_data(r_all[0, 7, index], r_all[1, 7, index])
    fig.canvas.draw()


stime.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    stime.reset()
button.on_clicked(reset)

plt.show()
