'''
EGEN KODE
'''

import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from challenge_C_part_4 import v_rad, phi_to_xy_transformation, v_rad_rel_home_star

seed = utils.get_seed('antonabr')
mission = SpaceMission(seed)

def spacecraft_position(measured_distances, known_positions):
    '''Function to calculate position using triangulation'''
    r1, r2, r3 = [measured_distances[i] for i in range(1,-2,-1)]    # unpacks distances from craft to planets/home star in AU
    pos1, pos2 = [known_positions[:,i] for i in range(1,-1,-1)]     # unpacks positions for home planet and neighbor planet rel. to home star in AU
    pos3 = np.array([0,0])                                          # home star position is in the origin in AU

    x1, y1 = pos1
    x2, y2 = pos2
    x3, y3 = pos3
    '''
    The following constants are the results from
    the two solution equations we got from the three
    circle equations which will intersect at our position.
    '''
    A = -2*(x1 - x2)
    B = -2*(y1 - y2)
    C = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2
    D = -2*(x1 - x3)
    E = -2*(y1 - y3)
    F = r1**2 - r3**2 - x1**2 + x3**2 - y1**2 + y3**2
    '''
    These are the solutions to the equations
    Ax+By=C
    Dx+Ey=F
    '''
    x = (C*E - B*F) / (A*E - B*D)   # x-position of the craft
    y = (A*F - C*D) / (A*E - B*D)   # y-position of the craft

    pos = np.array([x,y])           # in AU
    return pos
