import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from challenge_B import *
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from numba import njit
from scipy import interpolate



fig = plt.figure()
ax = fig.add_subplot()
ax.axis('off')

r_all = np.load('positions_all_planets.npy')
v_all = np.load('velocity_all_planets.npy')
time = np.load('time_planets.npy')

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)
mission = SpaceMission(seed)

M_sun = const.m_sun
AU = const.AU
yr = const.yr

initial_fuel_mass = 1e20 # kg
fuel_mass_loss = 9.203758666898709e19   # kg
initial_craft_mass = mission.spacecraft_mass + initial_fuel_mass
craft_mass = (initial_craft_mass - fuel_mass_loss) / M_sun
M_i = system.masses
G = const.G_sol # gravitational constant [AU^3/yr^2/m_sun]
M_star = system.star_mass   # [m_sun]
m = craft_mass

from_km_to_AU = 1 / AU
planet_radius = system.radii[0] * from_km_to_AU

plt.subplots_adjust(bottom=0.25)

N = 1000

l0, = plt.plot(r_all[0,0,::N], r_all[1,0,::N], lw=2, color='royalblue')
l1, = plt.plot(r_all[0,1,::N], r_all[1,1,::N], lw=2, color='orange')
l6, = plt.plot(r_all[0,6,::N], r_all[1,6,::N], lw=2, color='pink')

p0, = ax.plot(r_all[0,0,0], r_all[1,0,0], color='blue', marker='o', markersize=10)
p1, = ax.plot(r_all[0,1,0], r_all[1,1,0], color='darkorange', marker='o', markersize=10)
p6, = ax.plot(r_all[0,6,0], r_all[1,6,0], color='pink', marker='o', markersize=10)

p0end, = ax.plot([],[], color='k', marker='o', markersize=10)
p1end, = ax.plot([],[], color='k', marker='o', markersize=10)
p6end, = ax.plot([],[], color='k', marker='o', markersize=10)


craft_traj, = ax.plot(0,0, color='r')
craft_start, = ax.plot([],[], color='r', marker='o')

# ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

stime = Slider(axtime, 'Time [yr]', 0, 20)

def update(val):
    T = stime.val
    dt = time[1] - time[0]
    index = int(T / dt)
    # print(index)
    craft_position = spacecraft_position(dist, r_all[:,:,index])
    t, v_craft, r_craft = trajectory(time[index], craft_position, craft_velocity, 1.1, 0.001)
    index_ratio = len(time)*(t[-1] - T) / (time[-1]*len(t))         # Hvordan g?? fra indeks mellom rakett og planet
    last_index_rocket = len(t) - 1
    indexp = int(last_index_rocket*index_ratio)

    p0.set_data(r_all[0, 0, index], r_all[1, 0, index])
    p1.set_data(r_all[0, 1, index], r_all[1, 1, index])
    # p2.set_data(r_all[0, 2, index], r_all[1, 2, index])
    # p3.set_data(r_all[0, 3, index], r_all[1, 3, index])
    # p4.set_data(r_all[0, 4, index], r_all[1, 4, index])
    # p5.set_data(r_all[0, 5, index], r_all[1, 5, index])
    p6.set_data(r_all[0, 6, index], r_all[1, 6, index])
    # p7.set_data(r_all[0, 7, index], r_all[1, 7, index])
    craft_traj.set_data(r_craft[0,:], r_craft[1,:])
    craft_start.set_data(r_craft[0,0], r_craft[1,0])
    p0end.set_data(r_all[0, 0, index + indexp], r_all[1, 0, index + indexp])
    p1end.set_data(r_all[0, 1, index + indexp], r_all[1, 1, index + indexp])
    p6end.set_data(r_all[0, 6, index + indexp], r_all[1, 6, index + indexp])
    fig.canvas.draw()


stime.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    stime.reset()
button.on_clicked(reset)

plt.show()
