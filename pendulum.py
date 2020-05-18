import math
import enum
import numpy as np

import const

class Type(enum.IntEnum):
    POS = 0         # Position of the data
    VEL = 1         # Velocity of the data
    ACC = 2         # Acceletation of the data

globals().update(Type.__members__)

'''
Basic balence for the system, which set the cart position at x = 0
'''
def basic_balance():
    return 0


class pendulum():
    def __init__(self, M=0, m=0, L=0, mu_c=0.1, mu_p=0.01, signal=basic_balance):

        self.M = M                          # Mess of the cart
        self.m = m                          # Mess of the pendulum
        self.L = L                          # Length of the pendulum
        self.mu_c = mu_c                    # Friction of the cart
        self.mu_p = mu_p                    # Friction of the pole
        self.signal = signal                # Control signal function for the cart to move
        '''
            Angle of the pendulum on the cart,
            where theta = 0 is the normal of the cart
            3 dimension indicates the theta, theta', theta''
        '''
        self.theta = np.np.zeros(3)

        '''
            Position of the cart,
            where x = 0 is at the origin of x-axis
            3 dimension indicates the x, x', x''
        '''
        self.pos = np.np.zeros(3)
    
    '''
    '''
    def sys_react(self, f):

        total_mess = self.M + self.m

        #####################
        # Theta integration #
        #####################
        original_v = self.theta[Type.VEL]

        # Update velocity
        self.theta[Type.VEL] = self.theta[Type.VEL] + self.theta[Type.ACC] * const.period_t

        # Update position (Up+Low)*Height/2
        self.theta[Type.POS] = self.theta[Type.POS] + (original_v + self.theta[Type.VEL]) * const.period_t / 2

        self.theta[Type.ACC] = 0

        ########################
        # Position integration #
        ########################
        original_v = self.pos[Type.VEL]

        # Update velocity
        self.pos[Type.VEL] = self.pos[Type.VEL] + self.pos[Type.ACC] * const.period_t

        # Update position (Up+Low)*Height/2
        self.pos[Type.POS] = self.pos[Type.POS] + (original_v + self.pos[Type.VEL]) * const.period_t / 2

        self.pos[Type.ACC] = 0


        ######################
        # Theta acceleration #
        ######################
        # Numerator
        self.theta[Type.ACC] = self.theta[Type.ACC] + total_mess * const.g * math.sin(self.theta[Type.POS])

        self.theta[Type.ACC] = self.theta[Type.ACC] - math.cos(self.theta[Type.POS])*(f + self.m * self.L * self.theta[Type.VEL] * self.theta[Type.VEL] * math.sin(self.theta[Type.POS]) - total_mess * self.mu_c * np.sign(self.pos[Type.VEL]))

        self.theta[Type.ACC] = self.theta[Type.ACC] - self.mu_p * total_mess * self.theta[Type.VEL] / self.m / self.L

        # Denominator

        self.theta[Type.ACC] = self.theta[Type.ACC] / (4/3 * total_mess * self.L - self.m * self.L * math.pow( math.cos(self.theta[Type.POS]), 2))

        #########################
        # Position acceleration #
        #########################
        self.pos[Type.ACC] = self.pos[Type.ACC] + f

        self.pos[Type.ACC] = self.pos[Type.ACC] + self.m * self.L * (math.pow(self.theta[Type.VEL], 2) * math.sin(self.theta[Type.POS]) - self.theta[Type.ACC] * math.cos(self.theta[Type.POS]) )

        self.pos[Type.ACC] = self.pos[Type.ACC] / total_mess

        self.pos[Type.ACC] = self.pos[Type.ACC] - self.mu_c * np.sign(self.pos[Type.VEL])



    '''
    Applied a force to the inverted pendulum system
    '''
    def add_force(self, f):
        pass