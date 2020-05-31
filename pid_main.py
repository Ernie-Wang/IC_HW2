import pendulum                         # Control plant to optimize
import pso_v2 as pso                    # Optimizer algorithm
import pso_const as const               # Constant definition
import math
from matplotlib import pyplot as plt    # Data visualization
import numpy as np

# Weight for calculating fitness function
W = [88, 12, 2]
dim = 6

def fitness(theta, x, t):
    """ Fitness calculation """
    angle = abs(theta)
    shift = abs(x)
    time = t / const.TUNE_LIMIT
    return W[0] * angle + W[1] * shift + W[2] * time

def pid_sim(pid_param, plot_en=False):
    """
    Inverted pendulum simulation
    @param pid_param - the pid controller parameter
        # kp_x, ki_x, kd_x, kp_t, ki_t, kd_t
    @param plot_en - plot record data or not
        #note plot is blocking, you should close the figure manually and the program will continue
    """

    force = 0                               # Force last calculate

    # Create inverted pendulum system, signal set to origin point
    plant = pendulum.pendulum(M=const.M, m=const.m, L=const.L, mu_c=const.mu_c, mu_p=const.mu_p,signal=signal_const)
    plant.initial(const.theta_init, const.pos_init)
    
    # Create PID variable
    e_x = d_x = i_x = 0 # Position error term
    prev_ex = 0         # Previous position error
    i_x_bound = 10       # Prevent intergral overshoot
    e_t = d_t = i_t = 0 # Angle error term
    prev_et = 0         # Previous angle error
    i_t_bound = 3       # Prevent intergral overshoot

    pid_outmax = 80

    t = 0
    record_theta = []
    record_x = []
    desire_x = []
    record_x_e = []
    record_force = []
    for t in range(const.TUNE_LIMIT):
        # Calculate error for record
        x_d = plant.signal(t)
        error_x = plant.pos[0] - x_d

        # Judge if reach the controller sample time
        if t % const.REACT_T == 0:
            """ Control sample """
            e_x = plant.pos[0] - x_d            # position error
            d_x = e_x - prev_ex                 # position error differential
            i_x = i_x + e_x                     # position error integrate
            prev_ex = e_x                       # update previous position error

            e_t = plant.theta[0] - 0              # angle error
            d_t = e_t - prev_et                 # angle error differential
            i_t = i_t + e_t                     # angle error integrate
            prev_et = e_t                       # update previous angle error

            # Prevent position integral overshoot
            if i_x > i_x_bound:
                i_x = i_x_bound
            elif i_x < -1 * i_x_bound:
                i_x = -1 * i_x_bound

            # Prevent angle integral overshoot
            if i_t > i_t_bound:
                i_t = i_t_bound
            elif i_t < -1 * i_t_bound:
                i_t = -1 * i_t_bound

            pid_output = -1*(pid_param[0] * e_x + 0.01*pid_param[1] * i_x + pid_param[2] * d_x) +\
                            (pid_param[3] * e_t + 0.01*pid_param[4] * i_t + pid_param[5] * d_t)

            if pid_output > pid_outmax:
                pid_output = pid_outmax
            elif pid_output < -1 * pid_outmax:
                pid_output = -1 * pid_outmax

            force = pid_output

        # Terminate condition, if we are training, no ploting
        if not plot_en:
            # theta out of limit
            if abs(plant.theta[0]) >= const.theta_limit:
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
        
        # Run model
        # print(force)
        plant.add_force(force)
        record_theta.append(plant.theta[0])
        record_x.append(plant.pos[0])
        desire_x.append(x_d)
        record_x_e.append(error_x)
        record_force.append(force)

    len_record = len(record_theta)
    sample_data = 0
    sample_x_e = 0
    if len_record >= const.window:
        sample_data = record_theta[len_record - const.window:len_record].copy()
        sample_x_e = record_x_e[len_record - const.window:len_record].copy()
    else:
        sample_data = record_theta[:len_record].copy()
        sample_x_e = record_x_e[:len_record].copy()

    sample_data = np.abs(sample_data)
    sample_x_e = np.abs(sample_x_e)
    avg_theta = np.average(sample_data)
    avg_x_e = np.average(sample_x_e)

    fit = fitness(avg_theta, avg_x_e, t)

    if plot_en:
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        axs[0].plot(record_x, '--', desire_x, '-')
        axs[0].set_title('x')

        axs[1].plot(record_theta, '--')
        axs[1].set_title('theta')

        axs[2].plot(record_force, '--')
        axs[2].set_title('force')
        plt.show()

    return fit

def signal_const(t):
    """ Position signal generator """
    return 0

def simulate(pid_param):
    """ Normal simulation """
    return pid_sim(pid_param, True)

if __name__ == "__main__":
    # Initial PSO algorithm
    algo = pso.PSO (dim=dim, num=const.num, max_iter=const.max_iter, u_bound=const.p_range[1], l_bound=const.p_range[0], func=pid_sim, end_thres=const.end_thres, end_sample=const.end_sample, fit_max=const.fitness_max)

    # Initial particles
    algo.pso_init()

    # Run iteration
    algo.pso_iterator()

    # Extract best solution
    pid_param = algo.gbest.copy()

    # Simulate the result
    # pid_param = [0, 0., 0.00,
    #               0, 0.0,0]
    fit_value = simulate(pid_param)
    # plt.plot(algo.best_results)
    # plt.show()