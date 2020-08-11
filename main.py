##########################################
#  This code impliment fuzzy controller  #
##########################################

import pendulum
import const
import numpy as np
import abc_py
import pso_e
#import pso_v2 as pso_e                    # Optimizer algorithm
from matplotlib import pyplot as plt
#############################
#           GLOBAL          #
#############################

# coefficent for sliding mode surface, 
# c0 * theta + c1 * theta' + (c2 * x + c3 * x')(approach/depart)
# c4 => s coeficient feeding into fuzzy table
# c5 => s' coeficient feeding into fuzzy table
# [c0, c1, c2, c3, c4, c5]
C = [1, 1, 1, 1, 1, 1]

# Weight for calculating fitness function
W = [88, 10, 2]

# Dimension of the particle
dim = 6

# Desire position
desire_x = 0

'''
Calculate fitness function
fitness value = W_0 * abs(theta) + W_1 * (shift from x desire position) + W_2 * time
best solution is we have stable system, which theta is at desire angle, 
and x is at desire position, and use minimum time to became stable
'''
def fitness(theta, x, t):
    """
    @param theta - the angle between desire angle at the end
    @param x - shiftment from x desire at the end
    @param t - time
    @retval fitness value
    """
    angle = abs(theta)
    shift = abs(x)
    time = t / const.TUNE_LIMIT
    return W[0] * angle + W[1] * shift + W[2] * time

'''
Fuzzification, use to catagorize
'''
def fuzzification(v):
    shift_v = v + const.ZERO
    if shift_v >= 8:
        return 8
    elif shift_v <= 0:
        return 0
    else:
        return shift_v
        # return int(round(shift_v))


'''
Fuzzy simulation process for the certain coeficient c
'''
def fuzzy_sim(c, plot_en=False):
    """
    @param plot_en - plot record data or not
        #note plot is blocking, you should close the figure manually and the program will continue
    """
    s = 0                                           # Present s value
    last_s = 0                                      # last s value, use to calculate s'
    force = 0                                       # Force last calculate
    ad_mode = 1                                     # Departure or approaching mode, Depart = 1, approach = -1
    sys.initial(const.theta_init, const.pos_init)   # Initial pendulum system
    t = 0
    record_theta = []
    desire_theta = []
    record_x = []
    desire_x = []
    record_x_e = []
    for t in range(const.TUNE_LIMIT):
        #############################
        #      Calculate Force      #
        #############################

        # Judge approach or departure mode
        x_d = sys.signal(t)
        error_x = sys.pos[0] - x_d                          # e_x
        sign_e_x = np.sign(error_x)                         # sgn(e_x)
        sign_x_prom = np.sign(sys.pos[1])                   # sgn(x')
        sign_theta = np.sign(sys.theta[0])                  # sgn(theta)
        relation_x = sign_e_x * sign_x_prom                 # relation between e_x and x'
        relation_x_theta = sign_e_x * sign_theta            # Relation between e_x and theta
        if relation_x == 1 or relation_x_theta == 1:        # Departure mode
            ad_mode = 1
        else:                                               # Approaching mode
            ad_mode = -1


        # Calculate sliding surface value  
        last_s = s
        s = const.theta_scale * (c[0] * sys.theta[0] + c[1] * sys.theta[1]) + ad_mode * (c[2] * const.x_scale * error_x + c[3] * sys.pos[1])

        # Judge if reach the force threshold
        if t % const.REACT_T == 0:
            s_prom = s - last_s                     # s'
            weighted_s = c[4] * s                   # c4 * s
            weighted_s_prom = c[5] * s_prom         # c5 * s'

            # Catagory this status falled into 
            s_cata = fuzzification(weighted_s)              # s value catagorize
            s_prom_cata = fuzzification(weighted_s_prom)    # s' catagorize
            cata = const.FUZZY_TABLE[int(round(s_cata))][int(round(s_prom_cata))]
            norm_force = cata + (s_cata - round(s_cata) + s_prom_cata - round(s_prom_cata)) * 0.5 - const.ZERO

            # calculate the force to send, shift back to zero = 0
            force = norm_force / (const.SCALE - const.ZERO) * const.force_limit

        # Terminate condition, if we are training, no ploting
        if not plot_en:
            # theta out of limit
            if abs(sys.theta[0]) >= const.theta_limit:
                t = const.TUNE_LIMIT
                break
            
            # x out of limit
            if abs(error_x) >= const.x_limit:
                t = const.TUNE_LIMIT
                break

            # Theta is stable in a range of time
            len_record = len(record_theta)
            avg_theta = 0
            if len_record > const.window:
                sample_data = record_theta[len_record - int(const.window / 2):len_record].copy()
                sample_data = np.abs(sample_data)
                avg_theta = np.average(sample_data)
                if avg_theta < const.stable_theta:
                    break
        
        #############################
        #    Run pendulum model     #
        #############################
        # print(force)
        sys.add_force(force)

        # Record variable for further use
        record_theta.append(sys.theta[0])
        record_x.append(sys.pos[0])
        desire_x.append(x_d)
        record_x_e.append(error_x)

    len_record = len(record_theta)
    sample_data = 0
    sample_x_e = 0
    if len_record >= const.window:
        sample_data = record_theta[len_record - const.window:len_record].copy()
        sample_x_e = record_x_e[len_record - const.window:len_record].copy()
    else:
        sample_data = record_theta[:len_record].copy()
        sample_x_e = record_x_e[:len_record].copy()

    # Process data being record
    sample_data = np.abs(sample_data)
    sample_x_e = np.abs(sample_x_e)
    avg_theta = np.average(sample_data)
    avg_x_e = np.average(sample_x_e)

    # Calculate fitness value
    fit = fitness(avg_theta, avg_x_e, t)

    # Plot the whole process when the pendulum became stable
    if plot_en:
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        axs[0].plot(record_x, '--', desire_x, '-')
        axs[0].set_title('x')
        axs[1].plot(record_theta, '--')
        axs[1].set_title('theta')
        plt.show()

    return fit

'''
Simulation for the pendulum system
'''
def simulate(C):
    '''
    @param C - coefficent of sliding table
    '''
    return fuzzy_sim(C, True)

'''
Define square signal input
'''
triger = False
def signal_sqrt(t):
    global triger
    if ((t+1) % (10/const.PERIOD_T)) == 0:
        triger = not triger
    
    if triger:
        return -0.5
    return 0.5

'''
Set constant position
'''
def signal_const(t):
    return -4
if __name__ == "__main__":

    # Create inverted pendulum system, signal set to origin point
    # sys = pendulum.pendulum(M=const.M, m=const.m, L=const.L, mu_c=const.mu_c, mu_p=const.mu_p)

    # Create inverted pendulum system, signal set to constant desire position
    # sys = pendulum.pendulum(M=const.M, m=const.m, L=const.L, mu_c=const.mu_c, mu_p=const.mu_p,signal=signal_const)

    # Create inverted pendulum system, signal set square function
    sys = pendulum.pendulum(M=const.M, m=const.m, L=const.L, mu_c=const.mu_c, mu_p=const.mu_p,signal=signal_sqrt)

    sys.initial(const.theta_init, const.pos_init)
    # fit = simulate([11.88827976, 1.97008765, 14.13742648, 15.72209416, 1.38708104, 0.12893926]) # Test results

    # Initial ABC algorithm
    algo = abc_py.ABC (dim=dim, num=const.num, max_iter=const.max_iter, u_bound=const.p_range[1], l_bound=const.p_range[0], func=fuzzy_sim, end_thres=const.end_thres, end_sample=const.end_sample, fit_max=const.fitness_max)

    # Initial particles
    algo.abc_init()

    # Run iteration
    algo.abc_iterator()

    # Extract best solution
    C = algo.bestx.copy()


    # # Initial PSO algorithm
    # algo = pso_e.PSO (dim=dim, num=const.num, max_iter=const.max_iter, u_bound=const.p_range[1], l_bound=const.p_range[0], func=fuzzy_sim, end_thres=const.end_thres, end_sample=const.end_sample, fit_max=const.fitness_max)

    # # Initial particles
    # algo.pso_init()

    # # Run iteration
    # algo.pso_iterator()

    # # Extract best solution
    # C = algo.gbest.copy()


    # Simulate the result
    fit = simulate(C)
    plt.plot(algo.best_results)
    plt.show()
