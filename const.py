import math
import numpy as np

####################################################
#                       Time                       #
####################################################
# Time base is in million second, which equals to 0.001 second
cpu_cal_t = 1               # How many times does it take for a computer to calculate one 
                            # generation in algorithm, in million second

force_react_t = 20          # How long does it take for the force provider to react, in million second
period_t = force_react_t    # The time period for system to update, which is the same when the force is updated


####################################################
#                      System                      #
####################################################
M = 1.1
m = 0.2
L = 1
mu_c = 0.1
mu_p = 0.01
g = 9.8                     # The gravity constant for the system
force_limit = 80            # Limit of the force
theta_init = np.array([math.pi/6, 0, 0])
pos_init = np.array([0, 0, 0])


####################################################
#                      PSO/ABC                     #
####################################################
p_range = [-1, 1]           # Range of partition
gen = 50                    # Generation of one run in PSO



####################################################
#                       Fuzzy                      #
####################################################
'''
NB = 0
NM = 1
NS = 2
NZ = 2.5
Z  = 3
PZ = 3.5
PS = 4
PM = 5
PB = 6
'''
ZERO = 3

'''
FUZZY_TABLE[e][e']
'''
FUZZY_TABLE = np.array([[   3, 2.5,   2,   1,   0,   0,   0],
                        [ 3.5,   3, 2.5,   2,   1,   0,   0],
                        [   4, 3.5,   3, 2.5,   2,   1,   0],
                        [   5,   4, 3.5,   3, 2.5,   2,   1],
                        [   6,   5,   4, 3.5,   3, 2.5,   2],
                        [   6,   6,   5,   4, 3.5,   3, 2.5],
                        [   6,   6,   6,   5,   4, 3.5,   3]])