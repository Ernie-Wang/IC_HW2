import numpy as np
import csv
from matplotlib import pyplot as plt

from pso import PSO
from abc_py import ABC

''' Constant variable '''

######## Global variable #########
RUNS = 50
AGENT_NUM = 50
ITER_KINDS = 2
ALGO = 3
ITER = [500, 2500]
RESULTS = np.zeros((ALGO, RUNS, ITER_KINDS, ITER[1]))                   # Store all the result for the whole runs
LAST_ITER_AVG = np.zeros((ALGO, RUNS, ITER_KINDS))                   # Store all the result for the whole runs
AVERAGE_RESULT = np.zeros((ALGO, ITER_KINDS, ITER[1]))                   # Store all the result for the whole runs
init_best = np.zeros((ITER_KINDS))                   # Store all the result for the whole runs

##################################

######### PSO variable ###########

##################################

######### GSA variable ###########
epsilon = 1e-5
G_0 = 100
ALPHA = 20
K_best = 50
end_thres = 1e-5
##################################

######### ABC variable ###########

##################################
def write_file():
    filename = "./result_csv/Total_{func}.csv".format(func=test.name)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iter', 'PSO500', 'GSA500', 'ABC500', 'PSO2500', 'GSA2500', 'ABC2500'])
        writer.writerow([0, init_best[0], init_best[0], init_best[0], init_best[1], init_best[1], init_best[1]])
        for i in range(ITER[1]):
            if i < ITER[0]:
                writer.writerow([i+1, AVERAGE_RESULT[0][0][i], AVERAGE_RESULT[1][0][i], AVERAGE_RESULT[2][0][i], AVERAGE_RESULT[0][1][i], AVERAGE_RESULT[1][1][i], AVERAGE_RESULT[2][1][i]])
            else:
                writer.writerow([i+1, '', '', '', AVERAGE_RESULT[0][1][i], AVERAGE_RESULT[1][1][i], AVERAGE_RESULT[2][1][i]])
    pass

def plot_result():
    x1 = np.arange(0,  501,  1) 
    x2 = np.arange(0,  2501,  1) 
    plt.figure(1)
    # plt.subplot(3,  1,  1)  

    title = "{func}, ITER=500".format(func=test.name)
    plt.title(title) 
    plt.xlabel("iter") 
    plt.ylabel("fitness") 
    for i in range(ALGO):
        
        temp = AVERAGE_RESULT[i][0].copy()
        temp.resize(ITER[0], refcheck=False)
        temp = np.insert(temp, 1, init_best[0], 0)
        plt.plot(x1, temp) 
    plt.savefig("./result_csv/"+title)
    plt.figure(2)
    # plt.subplot(3,  1,  3)  

    title = "{func}, ITER=2500".format(func=test.name)
    plt.title(title) 
    plt.xlabel("iter") 
    plt.ylabel("fitness") 
    for i in range(ALGO):
        temp = AVERAGE_RESULT[i][1].copy()
        temp = np.insert(temp, 0, init_best[1])
        plt.plot(x2, temp)
    plt.savefig("./result_csv/" + title)
    plt.show()

def print_last_avg():
    result = np.mean(LAST_ITER_AVG, axis=1)
    result_mean = np.mean(result, axis=0)
    print("Last Iteration average, [ALGO][500/2500] \n", result)

def print_median():
    result_median = np.median(RESULTS, axis=1)
    ans = np.zeros((ALGO, ITER_KINDS))
    for i in range(ALGO):
        for j in range(ITER_KINDS):
            ans[i][j] = result_median[i][j][ITER[j]-1]
    print("Median value, [ALGO][500/2500] \n", ans)

def print_std():
    result_std = np.std(RESULTS, axis=1)
    ans = np.zeros((ALGO, ITER_KINDS))
    for i in range(ALGO):
        for j in range(ITER_KINDS):
            ans[i][j] = result_std[i][j][ITER[j]-1]
    print("std value, [ALGO][500/2500] \n", ans)

if __name__ == "__main__":
    for run in range(RUNS):
        for kind in range(ITER_KINDS):
            
            ## Initial random variables, every algorithm has same initial
            arr = np.random.uniform(test.l_bound,test.u_bound, (AGENT_NUM, test.dim))
            init_result = np.apply_along_axis(test.func, axis=1, arr=arr)
            best_idx = np.argmin(init_result)

            init_best[kind] = init_best[kind] + init_result[best_idx].copy()/RUNS

            #########   PSO   #########
            algo = PSO (dim=test.dim,num=AGENT_NUM,max_iter=ITER[kind], u_bound=test.u_bound, l_bound=test.l_bound, func=test.func, end_thres=end_thres)
            algo.pso_init(arr)
            algo.pso_iterator()

            # Resize the result to 2500
            tmp = algo.best_results.copy()
            tmp.resize(2500)
            RESULTS[0][run][kind] = tmp.copy()

            # Calculate mean fitness
            fitness_array = algo.get_current_fitness()
            LAST_ITER_AVG[0][run][kind] = np.mean(fitness_array)
            print(LAST_ITER_AVG[0][run][kind])
            ###########################


            #########   ABC   #########
            algo = ABC (dim=test.dim, num=AGENT_NUM, max_iter=ITER[kind], u_bound=test.u_bound, l_bound=test.l_bound, func=test.func, end_thres=end_thres)
            algo.abc_init(arr)
            algo.abc_iterator()

            # Resize the result to 2500
            tmp = algo.best_results.copy()
            tmp.resize(2500)
            RESULTS[2][run][kind] = tmp.copy()

            # Calculate mean fitness
            fitness_array = algo.get_current_fitness()
            LAST_ITER_AVG[2][run][kind] = np.mean(fitness_array)
            print(LAST_ITER_AVG[2][run][kind])
            ###########################

    for algo in range(ALGO):
        for kind in range(ITER_KINDS):

            average = np.zeros((ITER[1])) 
            for run in range(RUNS):
                
                average = average + RESULTS[algo][run][kind]
            AVERAGE_RESULT[algo][kind] = average / RUNS
    write_file()
    print_last_avg()
    print_median()
    print_std()
    plot_result()