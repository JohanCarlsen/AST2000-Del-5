'''
EGEN KODE
'''

import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission

seed = utils.get_seed('antonabr')
mission = SpaceMission(seed)

star_angles = mission.star_direction_angles
doppler_shifts_measured_at_sun = mission.star_doppler_shifts_at_sun

c = const.c_AU_pr_yr    # AU/yr
H_alpha = 656.3         # nm
deg_to_rad = np.pi / 180

# challenge C.1

def v_rad(delta_lambda, lambda_0 = H_alpha):
    '''
    Doppler effect
    Uses doppler shift as measured by reciever
    To find relative velocities with respect to
    the transmitter, the radial velocity changes
    direction, hence the minus sign
    '''
    v_r = c * -delta_lambda / lambda_0
    return v_r

sun_vr_rel_to_ref_stars = np.array([v_rad(doppler_shifts_measured_at_sun[0]), v_rad(doppler_shifts_measured_at_sun[1])])

def phi_to_xy_transformation(vector):
    '''Coordinate transformation'''
    phi_1 = star_angles[0] * deg_to_rad
    phi_2 = star_angles[1] * deg_to_rad
    transformation_array = np.array([[np.sin(phi_2), -np.sin(phi_1)], [-np.cos(phi_2), np.cos(phi_1)]])
    scaler = 1 / np.sin(phi_2 - phi_1)

    return scaler * np.matmul(transformation_array, vector)

def v_rad_rel_home_star(delta_lambda_1, delta_lambda_2, lambda_0 = H_alpha):
    '''
    This function takes to delta lambdas and
    calculates the radial velocities in both
    phi-systems. Then transform into xy-system.
    '''
    rad_vel_phi_system = np.array([v_rad(delta_lambda_1), v_rad(delta_lambda_2)])
    rad_vel_rel_to_sun_phi_system = rad_vel_phi_system - sun_vr_rel_to_ref_stars

    return phi_to_xy_transformation(rad_vel_rel_to_sun_phi_system)
