import math
import numpy as np

####################################################
#                       Time                       #
####################################################
# Time base is in million second, which equals to 0.001 second
CPU_CAL_T = 0.01                                # How many times does it take for a computer to calculate one 
                                                # generation in algorithm, in million second

REACT_T = 1                                     # How long does it take for the force provider to react, in million second
PERIOD_T = CPU_CAL_T                            # The time period for system to update, which is the same when the force is updated
MIN_LIMIT = 1.5                                 # How many minutes to run
TUNE_LIMIT = MIN_LIMIT * 60 / CPU_CAL_T         # The limit time for the simulation, minute * second per minute * millisecond per second


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
theta_limit = math.pi/3     # Limit of the theta
x_limit = 200               # Limit of x 
theta_init = np.array([math.pi/6, 0, 0])
pos_init = np.array([0, 0, 0])

end_thres = 1e-5            # Terminate threshold for the abc to terminate
end_sample = 20             # How many run we samples to judge the termination
max_iter = 1000             # Maximum iteration for abc


####################################################
#                      PSO/ABC                     #
####################################################
p_range = [0, 2]           # Range of partition
gen = 50                    # Generation of one run in PSO
num = 50



####################################################
#                       Fuzzy                      #
####################################################
'''
NB = 0
NM = 1
NS = 2
NZ = 3
Z  = 4
PZ = 5
PS = 6
PM = 7
PB = 8
'''
ZERO = 4                    # Value of zero, use as a base
SCALE = 8                   # The range of the fuzzy table is from 0~8

'''
FUZZY_TABLE[e][e']
'''
FUZZY_TABLE = np.array([[ 4, 3, 2, 1, 1, 1, 0, 0, 0],
                        [ 5, 4, 3, 2, 2, 2, 1, 0, 0],
                        [ 6, 5, 4, 3, 3, 3, 2, 1, 0],
                        [ 7, 6, 5, 4, 4, 4, 3, 2, 1],
                        [ 7, 6, 5, 4, 4, 4, 3, 2, 1],
                        [ 7, 6, 5, 4, 4, 4, 3, 2, 1],
                        [ 8, 7, 6, 5, 5, 5, 4, 3, 2],
                        [ 8, 8, 7, 6, 6, 6, 5, 4, 3],
                        [ 8, 8, 8, 7, 7, 7, 6, 5, 4]])