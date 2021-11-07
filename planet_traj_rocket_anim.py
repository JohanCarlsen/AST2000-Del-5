import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
# plt.xkcd()

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
planet6_radius = system.radii[6] * from_km_to_AU

# Kun en animasjonsfaktor som gjør det glattere
N = 1000

l0, = plt.plot(r_all[0,0,::N], r_all[1,0,::N], lw=2, color='royalblue')
l1, = plt.plot(r_all[0,1,::N], r_all[1,1,::N], lw=2, color='orange')
# l2, = plt.plot(r_all[0,2,::N], r_all[1,2,::N], lw=2)
# l3, = plt.plot(r_all[0,3,::N], r_all[1,3,::N], lw=2)
# l4, = plt.plot(r_all[0,4,::N], r_all[1,4,::N], lw=2)
# l5, = plt.plot(r_all[0,5,::N], r_all[1,5,::N], lw=2)
l6, = plt.plot(r_all[0,6,::N], r_all[1,6,::N], lw=2, color='violet')
# l7, = plt.plot(r_all[0,7,::N], r_all[1,7,::N], lw=2, color='gray')

p0, = ax.plot(r_all[0,0,0], r_all[1,0,0], color='blue', marker='o', markersize=10)
p1, = ax.plot(r_all[0,1,0], r_all[1,1,0], color='darkorange', marker='o', markersize=10)
# p2, = ax.plot(r_all[0,2,0], r_all[1,2,0], color='green', marker='o', markersize=6)
# p3, = ax.plot(r_all[0,3,0], r_all[1,3,0], color='darkred', marker='o', markersize=6)
# p4, = ax.plot(r_all[0,4,0], r_all[1,4,0], color='purple', marker='o', markersize=6)
# p5, = ax.plot(r_all[0,5,0], r_all[1,5,0], color='brown', marker='o', markersize=6)
p6, = ax.plot(r_all[0,6,0], r_all[1,6,0], color='violet', marker='o', markersize=10)
# p7, = ax.plot(r_all[0,7,0], r_all[1,7,0], color='gray', marker='o', markersize=10)
craft, = ax.plot([],[], color='r', marker='o')

T0 = 1.525           # Rundt 1.5 og 1.55 et sted ser morsomt ut (16.12-ish er også interessant) (1.52)
dt = time[1] - time[0]
index0 = int(T0 / dt)

craft_position = spacecraft_position(dist, r_all[:,:,index0])
t, v_craft, r_craft, r_i = trajectory(time[index0], craft_position, craft_velocity, 1.3, 0.001)
# print(time[index0])
ax.plot(r_craft[0], r_craft[1], 'r')
index_ratio = len(time)*(t[-1] - time[index0]) / (time[-1]*len(t))     # Deler på time[-1] fordi r_all er pr. 40 år, og time[-1] = 40, Hvordan gå fra indeks mellom rakett og planet
# print(index_ratio)
print('\ndistances [AU]:\n-----------------------')
def update(index):
    indexp = int(index_ratio*index)
    p0.set_data(r_all[0, 0, index0 + indexp], r_all[1, 0, index0 + indexp])
    p1.set_data(r_all[0, 1, index0 + indexp], r_all[1, 1, index0 + indexp])
    # p2.set_data(r_all[0, 2, index], r_all[1, 2, index])
    # p3.set_data(r_all[0, 3, index], r_all[1, 3, index])
    # p4.set_data(r_all[0, 4, index], r_all[1, 4, index])
    # p5.set_data(r_all[0, 5, index], r_all[1, 5, index])
    p6.set_data(r_all[0, 6, index0 + indexp], r_all[1, 6, index0 + indexp])
    # p7.set_data(r_all[0, 7, index], r_all[1, 7, index])
    craft.set_data(r_craft[0, index], r_craft[1, index])
    timelabel = ax.text(2, 3.2, f't={t[index] - time[index0]:.2f}yr', fontsize=16, weight='bold')
    lenp = np.sqrt(r_all[0, 6, index0 + indexp]**2 + r_all[1, 6, index0 + indexp]**2)
    lenr = np.sqrt(r_craft[0, index]**2 + r_craft[1, index]**2)
    if abs(lenr - lenp) <= 7.3e-4:        # 7e-4 AU er ca. Kármánlinjen
        print(f'\nplanet pos: {r_all[0, 6, index0 + indexp], r_all[1, 6, index0 + indexp]}, craft pos: {r_craft[0, index], r_craft[1, index]}')
        print(f'dist. diff: {abs(lenr - lenp)} at time {t[index] - time[index0]}')
        print(f'index, indexp: {index}, {indexp}')
    if lenr <= 1:
        print(f'{lenr} from the star')

    return p0, p1, p6, craft, timelabel

ani = FuncAnimation(fig, update, frames=range(0, len(time)), interval=1, blit=True)
plt.show()
