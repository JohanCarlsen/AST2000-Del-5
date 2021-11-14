import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission, LandingSequence
import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.shortcuts import SpaceMissionShortcuts

"""
EGEN KODE: Anton Brekke
"""

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)
mission = SpaceMission.load('mission_after_launch.pickle')
print('Look, I am still in space:', mission.rocket_launched)

'''Shortcut begin'''

code_stable_orbit = 95927
code_orientation = 43160
system = SolarSystem(seed)

shortcut = SpaceMissionShortcuts(mission, [code_stable_orbit, code_orientation])

# Orientation software shortcut
pos, vel, angle = shortcut.get_orientation_data()
print("Position after launch:", pos)
print("Velocity after launch:", vel)

#Verifying orientation with shortcut data
mission.verify_manual_orientation(pos, vel, angle)

# Initialize interplanetary travel instance
travel = mission.begin_interplanetary_travel()

# Shortcut to make the landing sequence class start with a stable orbit
shortcut.place_spacecraft_in_stable_orbit(2, 1000e3, 0, 6)

# Initializing landing sequence class instance
landing = mission.begin_landing_sequence()



def get_info_orbit(t, pos, vel):
    print('\n\n')
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    print(f't = {t}s eller {t / (60*60)}hr')
    print(f'r = {r}m')
    print(f'v = {v}m/s')
    # Regner ut radiell og tangentiell komponent etter vi er kommet i bane:
    # Merk, M >> m, så derfor er noen (m + M) = M og M*m / (M + m) = m
    vr = (pos[0]*vel[0] + pos[1]*vel[1]) / r
    vt = np.sqrt(v**2 - vr**2)
    print(f'v_r = {vr}m/s, v_t = {vt}m/s')

    M = system.masses[6] * const.m_sun
    m = mission.spacecraft_mass
    G = const.G
    print(f'Planet mass: {M}kg, spacecraft mass: {m}kg')

    Etot = 0.5*m*v**2 - G*m*M / r
    print(f'Total energy = {Etot}J')

    a = G*M*m / (2*abs(Etot))
    print(f'Major axis: {a}m')

    P = np.sqrt(4*np.pi**2*a**3 / (G*M))
    print(f'Orbital period: {P}s or {P / (60*60)}hr')

    b = r*vt * np.sqrt(a/(G*M))
    print(f'Semimajor axis:{b}m')

    e = np.sqrt(1 - (b/a)**2)
    print(f'Eccentricity: {e}')

    rA = a*(1 + e)
    rP = a*(1 - e)
    print(f'Apoapsis: {rA}m, Periapsis: {rP}m')
    info_array = np.array([r, vr, vt, a, b, e, P, rA, rP])
    print('\n\n')
    return info_array

# Calling landing sequece orient function
t, pos, vel = landing.orient()
info_1 = get_info_orbit(t, pos, vel)

landing.fall(9170 * 5)
t, pos, vel = landing.orient()
info_2 = get_info_orbit(t, pos, vel)

rel_error = abs(info_2 - info_1) / info_2
print(rel_error)
