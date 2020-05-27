import math
import numpy as np

####################################################
#                       Time                       #
####################################################
# Time base is in million second, which equals to 0.001 second
CPU_CAL_T = 0.001                                # How many times does it take for a computer to calculate one 
                                                # generation in algorithm, in million second

REACT_T = 1                                     # How long does it take for the force provider to react, in million second
PERIOD_T = CPU_CAL_T                            # The time period for system to update, which is the same when the force is updated
MIN_LIMIT = 1.5                                 # How many minutes to run
TUNE_LIMIT = int(MIN_LIMIT * 60 / CPU_CAL_T)         # The limit time for the simulation, minute * second per minute * millisecond per second


####################################################
#                      System                      #
####################################################
M = 1.1
m = 0.2
L = 1
mu_c = 0.1
mu_p = 0.01
g = 9.8                     # The gravity constant for the system
theta_scale = 80            # Scale for theta
x_scale = 8                 # Scale for theta
force_limit = 80            # Limit of the force
theta_limit = math.pi/3     # Limit of the theta
stable_theta = math.pi / 36 # definition of stable theta, set to 5 degree
x_limit = 200               # Limit of x 
theta_init = np.array([math.pi/6, 0, 0])
pos_init = np.array([0.00001, 0.0, 0.0])

fitness_max = 2.05          # Maximum fitness value for the lgorithm to terminate
end_thres = 1e-5            # Terminate threshold for the abc to terminate
end_sample = 30             # How many run we samples to judge the termination for algorithm
max_iter = 10             # Maximum iteration for abc
window = int(20 / PERIOD_T)      # Sample window for the theta


####################################################
#                      PSO/ABC                     #
####################################################
p_range = [0, 30]             # Range of partition
num = 50                        # Generation of one run in PSO