'''
EGEN KODE
'''

import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from numba import njit
from scipy import interpolate
from challenge_A import trajectory

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)
mission = SpaceMission(seed)

G_sol = const.G_sol
planet_mass = system.masses[6]
M_star = system.star_mass
axis = system.semi_major_axes[6]
P = np.sqrt(4 * np.pi**2 * axis**3 / (G_sol * (M_star + planet_mass)))
AU = const.AU / 1000 # [km]
home_planet_R = system.radii[0] / AU # [AU]

r_all = np.load('positions_all_planets.npy')
t_all = np.load('time_planets.npy')

P_planet_index = np.logical_and(t_all >= P-1e-4, t_all <= P+1e-4)
index_P = np.where(P_planet_index == True)[0][1]

dt = t_all[1]

r_planets = np.zeros((2,3,index_P))
r_planets[:,0,:] = r_all[:,0,:index_P]
r_planets[:,1,:] = r_all[:,1,:index_P]
r_planets[:,2,:] = r_all[:,6,:index_P]

dist_diff = np.sum((r_planets[:,0,:] - r_planets[:,2,:])**2, axis=0)
min_dist_index = np.where(dist_diff == np.min(dist_diff))[0][0]
min_dist_time = min_dist_index * dt

# print(min_dist_time)
'''
0.39681963099402645
'''

plt.plot(r_planets[0,0,:], r_planets[1,0,:])
plt.plot(r_planets[0,1,:], r_planets[1,1,:])
plt.plot(r_planets[0,2,:], r_planets[1,2,:])
plt.plot(r_planets[0,0,min_dist_index], r_planets[1,0,min_dist_index], 'ro')
plt.plot(r_planets[0,1,min_dist_index], r_planets[1,1,min_dist_index], 'ro')
plt.plot(r_planets[0,2,min_dist_index], r_planets[1,2,min_dist_index], 'ro')
#
# plt.show()

home_planet_vec = r_planets[:,0,min_dist_index]
unit_r = home_planet_vec / np.linalg.norm(home_planet_vec)

t_launch = min_dist_time # [yr]
position_after_launch = np.array([[1.85891332, 1.17687184e-4]])
r_0 = home_planet_vec + (np.array([home_planet_R, 0]) * unit_r)
v_after_launch = np.array([0, 6.6])

t, v_craft, r_craft = trajectory(t_launch, r_0, v_after_launch, 1.55, 0.001)
print(r_craft[:,0])
# plt.plot(r_craft[0,:], r_craft[1,:])
# plt.axis('equal')
# plt.show()
