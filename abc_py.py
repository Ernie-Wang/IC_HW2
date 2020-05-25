
import numpy as np
import random


end_thres = 1e-5
class ABC():

    def __init__(self, dim, num, max_iter, u_bound, l_bound, func, end_thres, end_sample, fit_max):
        """ Initialize ABC object """
        self.SN = num                                 # Number of onlooker bees / enployed bees
        self.dim = dim                                # Searching dimension
        self.limit = 0.2*num                          # Searching limit
        self.max_iter = max_iter                      # Maximum iteration
        self.u_bound = u_bound                        # Upper bound
        self.l_bound = l_bound                        # Lower bound
        self.func = func                              # Benchmark function
        self.end_thres = end_thres                    # Terminate threshold
        self.end_sample = end_sample                  # End sample number
        self.fit_max = fit_max                        # Maximum fitness value

        self.X = np.zeros((self.SN, self.dim))        # Food source position
        self.fit = np.zeros((self.SN))                # Food source fitness
        self.trial = np.zeros((self.SN))              # Food source try time
        self.bestx = np.zeros((self.dim))             # Global best position
      
        self.best = 1000                              # Global best fitness
        self.best_results = np.zeros((self.max_iter)) # Fitness value of the agent

    def softmax(self, arr):
        nm = np.linalg.norm(arr)
        if nm == 0:
          norm1 = arr / arr.size
        else:
          norm1 = arr / nm
        inv_arr = -1 * norm1
        total = np.sum(np.exp(inv_arr))
        return np.exp(inv_arr)/total

    def triger(self, iteration):
        upper = lower = self.best_results[iteration]
        # if  iteration > self.end_sample:
        if self.best_results[iteration] < self.fit_max and iteration > self.end_sample:
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

    def abc_init(self, X=None):
        # Initialize food source for all employed bees
        if X is None:
            for i in range(self.SN):
                self.X[i] = np.random.uniform(self.l_bound,self.u_bound, (self.dim))
                self.fit[i] = self.func(self.X[i])
        else:
            if X.shape == self.X.shape:
               self.X = X.copy()
               self.fit = np.apply_along_axis(self.func, axis=1, arr=self.X)
            else:
                raise Exception("Custom data shape error")

        best_idx = np.argmin(self.fit)
        self.best = self.fit[best_idx]

    def abc_iter(self, ite_idx):
        print("Iteration: {ite}, best is {best:6.3f}, best C = {C_best}".format(ite=ite_idx+1, best=self.best, C_best=self.bestx))

        # 1. Send employed bees to the new food source
        for i in range(self.SN):
            # Random select a source to change but noice that j!= i
            j = random.choice([n for n in range(self.SN) if i != n])
            tmp_pos = self.X[i] + np.random.uniform(-1,1,(self.dim)) * (self.X[i] - self.X[j])
            tmp_pos = np.where(tmp_pos > self.u_bound, self.u_bound, tmp_pos)
            tmp_pos = np.where(tmp_pos < self.l_bound, self.l_bound, tmp_pos)

            # Calculate new position fitness
            tmp_fit = self.func(tmp_pos)
            # Greedy selection to select food source
            if tmp_fit < self.fit[i]:
                self.X[i] = tmp_pos
                self.fit[i] = tmp_fit
            else:
                self.trial[i] = self.trial[i] + 1
                

        # Calculate Selection Probabilities
        p_source = self.softmax(self.fit)
        # 2. Send onlooker bees
        for i in range(self.SN):
            # Select Source Site by Roulette Wheel Selection)
            food_source = np.random.choice(self.SN, 1, p=p_source)
            # Random select a source to change but noice that j!= i
            j = random.choice([n for n in range(self.SN) if i != n])
            tmp_pos = self.X[i] + np.random.uniform(-1,1,(self.dim)) * (self.X[i] - self.X[j])
            tmp_pos = np.where(tmp_pos > self.u_bound, self.u_bound, tmp_pos)
            tmp_pos = np.where(tmp_pos < self.l_bound, self.l_bound, tmp_pos)
            # Calculate new food source fitness
            tmp_fit = self.func(tmp_pos)
            # Greedy selection to select food source
            if tmp_fit < self.fit[food_source]:
                self.X[food_source] = tmp_pos
                self.fit[food_source] = tmp_fit
            else:
                self.trial[food_source] = self.trial[food_source] + 1

        # 3. Send scout
        for i in range(self.SN):
            if self.trial[i] >= self.limit:
                self.trial[i] = 0
                self.X[i] = np.random.uniform(self.l_bound,self.u_bound, (self.dim))
                self.fit[i] = self.func(self.X[i])
        
        # 4. Update best solution
        best_idx = np.argmin(self.fit)
        if self.fit[best_idx] < self.best:
            self.best = self.fit[best_idx]
            self.bestx = self.X[best_idx].copy()
        
        self.best_results[ite_idx] = self.best


    def abc_iterator(self):
        # Iteration
        for ite_idx in range(self.max_iter):
            self.abc_iter(ite_idx)

            if self.triger(ite_idx):
                break

    def get_current_fitness(self):
        """ Get current fitness of each agent """
        return np.apply_along_axis(self.func, axis=1, arr=self.X)

if __name__ == "__main__":
    a = ABC (dim=test.dim, num=50, max_iter=2500, u_bound=test.u_bound, l_bound=test.l_bound, func=test.func, end_thres=end_thres)
    arr = np.random.uniform(test.l_bound,test.u_bound, (50, test.dim))
    a.abc_init(arr)
    a.abc_iterator()

    # Calculate mean fitness
    fitness_array = a.get_current_fitness()
    mean_fitness = np.mean(fitness_array)
    print(mean_fitness)