""" Python PSO """
import random
import numpy as np


end_thres = 1e-5
class PSO():
    def __init__(self, dim, num, max_iter, u_bound, l_bound, func, end_thres, end_sample, fit_max):
        """ Initialize PSO object """
        self.dim = dim                                # Searching dimension
        self.num = num                                # Number of particle
        self.max_iter = max_iter                      # Maximum iteration
        self.X = np.zeros((self.num, self.dim))       # Particle position
        self.V = np.zeros((self.num, self.dim))       # Particle velocity
        self.pbest = np.zeros((self.num, self.dim))   # Pbest position
        self.pbest_v = np.zeros(self.num)             # Pbest value
        self.gbest = np.zeros((1, self.dim))          # Gbest position
        self.gbest_v = 1000                           # Gbest value
        self.u_bound = u_bound                        # Upper bound
        self.l_bound = l_bound                        # Lower bound
        self.func = func                              # Benchmark function
        self.end_thres = end_thres                    # Terminate threshold
        self.end_sample = end_sample                  # End sample number
        self.fit_max = fit_max                        # Maximum fitness value

        self.best_results = np.zeros((self.max_iter))                   # Fitness value of the agent

    def triger(self, iteration):
        upper = lower = self.best_results[iteration]
        if  iteration > self.end_sample:
        # if self.best_results[iteration] < self.fit_max and iteration > self.end_sample:
            for i in range(self.end_sample):
                if upper < self.best_results[iteration - i]:
                    upper = self.best_results[iteration - i]

                elif lower > self.best_results[iteration - i]:
                    lower = self.best_results[iteration - i]
            if(upper-lower) < self.end_thres:
                self.best_results[iteration:] = self.best_results[iteration]
                return True
            else:
                return False

    def pso_init(self, X=None):
        """ Initialize particle attribute, best position and best value """
        if X is None:
            for n in range(self.num):
                for d in range(self.dim):
                    self.X[n][d] = random.uniform(self.l_bound,self.u_bound)
                    self.V[n][d] = random.uniform(-1,1)
                self.pbest[n] = self.X[n].copy()
                self.pbest_v[n] = self.func(self.pbest[n])
        else:
            if X.shape == self.X.shape:
                self.X = X.copy()
                self.V = np.random.uniform(self.l_bound,self.u_bound, (self.num, self.dim))
                self.pbest = self.X.copy()
                self.pbest_v = np.apply_along_axis(self.func, axis=1, arr=self.X)
            else:
                raise Exception("Custom data shape error")

        best_idx = np.argmin(self.pbest_v)
        self.gbest_v = self.pbest_v[best_idx]
        self.gbest = self.pbest[best_idx].copy()

    def pso_iterator(self):
        """ Iteration """
        for ite_idx in range(self.max_iter):
            print("Iteration: {ite}, best is {best:6.3f}, best C = {C_best}".format(ite=ite_idx+1, best=self.gbest_v, C_best=self.gbest))  
            # print("Iteration: {ite}, best is {best}".format(ite=ite_idx+1, best=self.gbest_v))
            
            # Update particle position and velocity
            r1 = np.random.uniform(size=(self.num, self.dim))
            r2 = np.random.uniform(size=(self.num, self.dim))
            self.V = self.V*random.uniform(0.2,0.6) + 2*r1*(self.pbest-self.X) + 2*r2*(self.gbest-self.X)
            tmp_X = self.X + self.V
            tmp_X = np.where(tmp_X > self.u_bound, self.u_bound, tmp_X)
            tmp_X = np.where(tmp_X < self.l_bound, self.l_bound, tmp_X)
            self.X = tmp_X.copy()

            # Particle iterator, update best value
            for part in range(self.num):
                test_tmp = self.func(self.X[part])
                # Update local attribute
                if test_tmp < self.pbest_v[part]:
                    self.pbest[part] = self.X[part].copy()
                    self.pbest_v[part] = test_tmp

                    # Update global attribute
                    if test_tmp < self.gbest_v:
                        self.gbest = self.X[part].copy()
                        self.gbest_v = test_tmp

            # print(self.gbest)
            self.best_results[ite_idx] = self.gbest_v

            self.func(self.gbest)
            if self.triger(ite_idx):
                break

    def get_current_fitness(self):
        """ Get current fitness of each agent """
        return np.apply_along_axis(self.func, axis=1, arr=self.X)

if __name__ == "__main__":
    a = PSO (dim=test.dim, num=50,max_iter=2500, u_bound=test.u_bound, l_bound=test.l_bound, func=test.func, end_thres=end_thres)
    arr = np.random.uniform(test.l_bound,test.u_bound, (50, test.dim))
    a.pso_init(arr)
    a.pso_iterator()

    # Calculate mean fitness
    fitness_array = a.get_current_fitness()
    mean_fitness = np.mean(fitness_array)
    print(mean_fitness)