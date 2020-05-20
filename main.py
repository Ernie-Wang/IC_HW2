import pendulum
import const
import numpy as np
import abc_py

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
W = [10, 9, 2]

# Dimension of the particle
dim = 6

# Desire position
desire_x = 0

'''
One iteration for the fuzzy loop
'''
def fitness(theta, x, t):
    angle = abs(theta[0])
    shift = abs(x[0])
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
        return int(round(shift_v))


'''
Fuzzy simulation process for the certain coeficient c
'''
def fuzzy_sim(c):
    s = 0                                   # Present s value
    last_s = 0                              # last s value, use to calculate s'
    force = 0                               # Force last calculate
    ad_mode = 1                             # Departure or approaching mode, Depart = 1, approach = -1
    sys.initial(const.theta_init, const.pos_init)
    for t in range(const.TUNE_LIMIT):

        # Run model
        # print(force)
        sys.add_force(force)

        # Judge approach or departure mode
        error_x = sys.pos[0] - sys.signal(t)                # e_x
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
        s = c[0] * sys.theta[0] + c[1] * sys.theta[1] + ad_mode * (c[2] * error_x + c[3] * sys.pos[1])

        # Judge if reach the force threshold
        if t % const.REACT_T == 0:
            s_prom = s - last_s                     # s'
            weighted_s = c[4] * s                   # c4 * s
            weighted_s_prom = c[5] * s_prom         # c5 * s'

            # Catagory this status falled into 
            s_cata = fuzzification(weighted_s)              # s value catagorize
            s_prom_cata = fuzzification(weighted_s_prom)    # s' catagorize
            cata = const.FUZZY_TABLE[s_cata][s_prom_cata]

            # calculate the force to send, shift back to zero = 0
            force = ( cata - const.ZERO ) / const.SCALE * const.force_limit

        # Terminate condition
        # theta out of limit
        if abs(sys.theta[0]) >= const.theta_limit:
            break
        
        # x out of limit
        if abs(sys.pos[0]) >= const.x_limit:
            break

    fit = fitness(sys.theta, sys.pos, t)
    
    return fit

'''
Simulation for the pendulum system
'''
def simulate(C):
    fuzzy_sim(C)


if __name__ == "__main__":
    # Create inverted pendulum system
    sys = pendulum.pendulum(M=const.M, m=const.m, L=const.L, mu_c=const.mu_c, mu_p=const.mu_p)
    sys.initial(const.theta_init, const.pos_init)

    # Initial ABC algorithm
    algo = abc_py.ABC (dim=dim, num=const.num, max_iter=const.TUNE_LIMIT, u_bound=const.p_range[1], l_bound=const.p_range[0], func=fuzzy_sim, end_thres=const.theta_limit)
    
    # Initial particles
    algo.abc_init()

    # Run iteration
    algo.abc_iterator()

    # Extract best solution
    C = algo.bestx.copy()

    # Simulate the result
    simulate(C)
