'''
EGEN KODE
'''

import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from numba import njit
from scipy import interpolate
from challenge_A import trajectory
from challenge_C_part_4 import v_rad, phi_to_xy_transformation, v_rad_rel_home_star
from challenge_D_part_4 import spacecraft_position

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

dt = t_all[1] - t_all[0]

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

''' shortcut from part 1'''
code_engine = 50557
code_launch = 25206
code_escape_trajectory = 14143
shortcut = SpaceMissionShortcuts(mission, [code_engine, code_launch, code_escape_trajectory])

number_density = 1e5 / (1e-6)**3
temperature = 5e3
hole_area = 0.25 * 1e-6**2

thrust_pr_box, mass_loss_pr_box = shortcut.compute_engine_performance(number_density, temperature, hole_area)

N_box = 2e30
initial_fuel_mass = 1e20 # kg
est_launch_dur = 500 # s
thrust = thrust_pr_box * N_box
mass_loss_rate = mass_loss_pr_box * N_box

time_of_launch = 0.39681963099402645
launch_index = int(np.ceil(time_of_launch/dt))
r_p_vec = r_all[:,0,launch_index]
r_p_unit = r_p_vec / np.linalg.norm(r_p_vec)
radius_vec = home_planet_R * r_p_unit
r_0 = r_p_vec + radius_vec

mission.set_launch_parameters(thrust, mass_loss_rate, initial_fuel_mass, est_launch_dur, r_0, time_of_launch)
mission.launch_rocket()
shortcut.get_launch_results()

height_above_suface = 1e7 # m
direction = 90 # degrees wrt. x-axis
fuel_left = 1e4 # kg

shortcut.place_spacecraft_on_escape_trajectory(thrust, mass_loss_rate, time_of_launch, height_above_suface, direction, fuel_left)
''' shortcut end '''


delta_lambda_1, delta_lambda_2 = mission.measure_star_doppler_shifts()
craft_velocity = v_rad_rel_home_star(delta_lambda_1, delta_lambda_2)
dist = mission.measure_distances()
craft_position = spacecraft_position(dist, r_all[:,:,launch_index])
# mission.verify_manual_orientation(craft_position, craft_velocity, 0)

if __name__ == '__main__':
    t, v_craft, r_craft, r_i = trajectory(time_of_launch, craft_position, craft_velocity, 1, 0.001)

    plt.plot(r_i[0,0,:], r_i[1,0,:])
    plt.plot(r_i[0,1,:], r_i[1,1,:])
    plt.plot(r_i[0,2,:], r_i[1,2,:])

    plt.plot(r_i[0,0,0], r_i[1,0,0], 'ro')
    plt.plot(r_i[0,1,0], r_i[1,1,0], 'bo')
    plt.plot(r_i[0,2,0], r_i[1,2,0], 'go')
    plt.plot(r_i[0,0,-1], r_i[1,0,-1], 'ro')
    plt.plot(r_i[0,1,-1], r_i[1,1,-1], 'bo')
    plt.plot(r_i[0,2,-1], r_i[1,2,-1], 'go')
    plt.plot([0,0], [0,0], 'ko')

    plt.plot(r_craft[0,:], r_craft[1,:])

    plt.show()
