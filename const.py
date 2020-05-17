
# Time base is in million second, which equals to 0.001 second

cpu_cal_t = 1               # How many times does it take for a computer to calculate one 
                            # generation in algorithm, in million second

force_react_t = 20          # How long does it take for the force provider to react, in million second
period_t = force_react_t    # The time period for system to update, which is the same when the force is updated

g = 9.8                     # The gravity constant for the system