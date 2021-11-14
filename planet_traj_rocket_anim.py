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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fig = plt.figure()
ax = fig.add_subplot()
ax.axis('off')
# plt.xkcd()

r_all = np.load('positions_all_planets.npy')
v_all = np.load('velocity_all_planets.npy')
time = np.load('time_planets.npy')

rocket_img = plt.imread('rocket.jpg')
planet_1_img = plt.imread('planet_1.png')
planet_2_img = plt.imread('planet_2.png')
planet_3_img = plt.imread('planet_3.png')
image_box_rocket = OffsetImage(rocket_img, zoom=0.07)
image_box_planet_1 = OffsetImage(planet_1_img, zoom=0.1)
image_box_planet_2 = OffsetImage(planet_2_img, zoom=0.15)
image_box_planet_3 = OffsetImage(planet_3_img, zoom=0.08)

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
print(system.radii[6])
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
craft, = ax.plot([],[], color='r', marker='o', markersize=8)

T0 = 1.5259           # Rundt 1.5 og 1.55 et sted ser morsomt ut (16.12-ish er også interessant) (1.5259)
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
    if index == 0:
        input()         # Dette er kun for å gi meg tid til å ta opptak av animasjonen
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
    diff_vec = r_all[:, 6, index0 + indexp] - r_craft[:, index]
    diff_norm = np.linalg.norm(diff_vec)
    lenr = np.sqrt(r_craft[0, index]**2 + r_craft[1, index]**2)
    l = lenr * np.sqrt(M_i[6] / (10*M_star))
    rocket = AnnotationBbox(image_box_rocket, (r_craft[0, index], r_craft[1, index]), frameon=False)
    planet_1 = AnnotationBbox(image_box_planet_1, (r_all[0, 0, index0 + indexp], r_all[1, 0, index0 + indexp]), frameon=False)
    planet_2 = AnnotationBbox(image_box_planet_2, (r_all[0, 1, index0 + indexp], r_all[1, 1, index0 + indexp]), frameon=False)
    planet_3 = AnnotationBbox(image_box_planet_3, (r_all[0, 6, index0 + indexp], r_all[1, 6, index0 + indexp]), frameon=False)
    ax.add_artist(rocket)
    ax.add_artist(planet_1)
    ax.add_artist(planet_2)
    ax.add_artist(planet_3)
    if diff_norm <= 1e-2 + planet6_radius:        # 6.7e-5 AU er ca. 10 000km
        print(f'\ndistance for orbital maneuver: {l}')
        print(f'planet pos: {r_all[0, 6, index0 + indexp], r_all[1, 6, index0 + indexp]}, craft pos: {r_craft[0, index], r_craft[1, index]}')
        print(f'planet vel: {v_all[0, 6, index0 + indexp], v_all[1, 6, index0 + indexp]}, craft pos: {v_craft[0, index], v_craft[1, index]}')
        print(f'dist. diff: {diff_norm} at time {t[index] - time[index0]} after launch, time {t[index]} in total.')
        print(f'index, indexp: {index}, {indexp}')
        print(f'Orbital maneuver possible: {diff_norm <= l}')
    if lenr <= 1:
        print(f'{lenr} from the star')

    return p0, p1, p6, craft, timelabel, planet_1, planet_2, planet_3, rocket

ani = FuncAnimation(fig, update, frames=range(0, len(time)), interval=1, blit=True)
plt.show()
dist = np.zeros((len(r_craft[0,:]), 1))
for index in range(len(r_craft[0,:])):
    indexp = int(index_ratio*index)
    dist[index] = np.linalg.norm(r_craft[:, index] - r_all[:, 6, index0 + indexp])

print(np.min(dist))
index = np.where(dist==np.min(dist))[0][0]
print(index)
lenr = np.sqrt(r_craft[0, index]**2 + r_craft[1, index]**2)
l = lenr * np.sqrt(M_i[6] / (10*M_star))
print(f'Maneuver possible: {np.min(dist) <= l}')
