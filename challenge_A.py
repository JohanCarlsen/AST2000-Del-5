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

r_all = np.load('positions_all_planets.npy')  # array containing all planet positions
t_all = np.load('time_planets.npy') # array containing the time for planet orbits

def trajectory(inital_time, initial_position, initial_velocity, simulation_time, time_step_length):

    dt = time_step_length
    time_steps = int(np.ceil(simulation_time / dt))
    t = np.linspace(inital_time, inital_time + simulation_time, time_steps)

    inter_func = interpolate.interp1d(t_all, r_all, axis=-1)
    r_i = inter_func(t)

    r_craft = np.zeros((2,time_steps))
    v_craft = np.zeros((2,time_steps))

    r_craft[:,0] = initial_position
    v_craft[:,0] = initial_velocity

    for i in range(time_steps-1):
        r_vec = r_craft[:,i]
        r_vec_norm = np.linalg.norm(r_vec)
        r_i_vec = r_i[:,:,i]

        F_star = -G * m * M_star * r_vec / r_vec_norm**3
        F_planets = 0

        for j in range(8):
            r_planet_i_vec = r_i_vec[:,j]
            r_r_i = r_vec - r_planet_i_vec
            r_r_i_norm = np.linalg.norm(r_r_i)

            F_planets += G * m * M_i[j] / r_r_i_norm**3 * r_r_i

        F_tot = F_star - F_planets
        a = F_tot / m

        v_craft[:,i+1] = v_craft[:,i] + a*dt
        r_craft[:,i+1] = r_craft[:,i] + v_craft[:,i+1]*dt

    return t, v_craft, r_craft
#
# test_pos = np.array([20, -10])
# test_vel = np.array([-10,10])
# t, v_craft, r_craft = trajectory(0,test_pos, test_vel, 15, 0.001)
#
# plt.subplot(311)
# plt.plot(t, r_craft.T)
# plt.subplot(312)
# plt.plot(t, v_craft.T)
# plt.subplot(313)
# plt.plot(r_craft[0,:], r_craft[1,:])
# plt.show()
