import numpy as np

# import tqdm import tnrange for progress bar
from tqdm import trange

def test():

    def fitness(binary_gene):
        w = np.array([71, 34, 82, 23, 1, 88, 12, 57, 10, 68, 5, 33, 37, 69, 98, 24, 26, 83, 16, 26, 18, 43, 52, 71, 22, 65, 68, 8, 40, 40, 24, 72, 16, 34, 10, 19, 28, 13, 34, 98, 29, 31, 79, 33, 60, 74, 44, 56, 54, 17, 63, 83, 100, 54, 10, 5, 79, 42, 65, 93, 52, 64, 85, 68, 54, 62, 29, 40, 35, 90, 47, 77, 87, 75, 39, 18, 38, 25, 61, 13, 36, 53, 46, 28, 44, 34, 39, 69, 42, 97, 34, 83, 8, 74, 38, 74, 22, 40, 7, 94])
        v = np.array([26, 59, 30, 19, 66, 85, 94, 8, 3, 44, 5, 1, 41, 82, 76, 1, 12, 81, 73, 32, 74, 54, 62, 41, 19, 10, 65, 53, 56, 53, 70, 66, 58, 22, 72, 33, 96, 88, 68, 45, 44, 61, 78, 78, 6, 66, 11, 59, 83, 48, 52, 7, 51, 37, 89, 72, 23, 52, 55, 44, 57, 45, 11, 90, 31, 38, 48, 75, 56, 64, 73, 66, 35, 50, 16, 51, 33, 58, 85, 77, 71, 87, 69, 52, 10, 13, 39, 75, 38, 13, 90, 35, 83, 93, 61, 62, 95, 73, 26, 85])
        w_ = np.sum(w * binary_gene)
        if w_ > 1000:
    #         print(np.sum(w * arr))
            return 1.0 / (w_ - 1000)
        else:
            return np.sum(v * binary_gene)

    num_pop = 30
    problem_dimentions = 100

    test = BGA(pop_shape=(num_pop, problem_dimentions), method=fitness, p_c=0.8, p_m=0.2, max_round = 1000, early_stop_rounds=None, verbose = None, maximum=True)
    best_solution, best_fitness = test.run()
    print(test.generations)


class BGA():
    """
    Simple 0-1 genetic algorithm.
    User Guide:
    >> test = GA(pop_shape=(10, 10), method=np.sum)
    >> solution, fitness = test.run()
    """
    def __init__(self, pop_shape, method, p_c=0.8, p_m=0.2, max_round = 1000, early_stop_rounds=None, verbose = None, maximum=True):
        """
        Args:
            pop_shape: The shape of the population matrix.
            method: User-defined medthod to evaluate the single individual among the population.
                    Example:

                    def method(arr): # arr is a individual array
                        return np.sum(arr)

            p_c: The probability of crossover.
            p_m: The probability of mutation.
            max_round: The maximun number of evolutionary rounds.
            early_stop_rounds: Default is None and must smaller than max_round.
            verbose: 'None' for not printing progress messages. int type number for printing messages every n iterations.
            maximum: 'True' for finding the maximum value while 'False' for finding the minimum value.
        """
        if early_stop_rounds != None:
            # assert refers to always true statemetns in code.
            assert(max_round > early_stop_rounds)
        self.pop_shape = pop_shape
        self.method = method
        self.pop = np.zeros(pop_shape)
        self.fitness = np.zeros(pop_shape[0])
        self.p_c = p_c
        self.p_m = p_m
        self.max_round = max_round
        self.early_stop_rounds = early_stop_rounds
        self.verbose = verbose
        self.maximum = maximum

        # self.generations = []

    def evaluation(self, pop):
        """
        Computing the fitness of the input popluation matrix.
        Args:
            p: The population matrix need to be evaluated.
        """
        return np.array([self.method(i) for i in pop])

    def initialization(self):
        """
        Initalizing the population which shape is self.pop_shape(0-1 matrix).
        """
        self.pop = np.random.randint(low=0, high=2, size=self.pop_shape)
        self.fitness = self.evaluation(self.pop)

    def crossover(self, ind_0, ind_1):
        """
        Single point crossover.
        Args:
            ind_0: individual_0
            ind_1: individual_1
        Ret:
            new_0, new_1: the individuals generatd after crossover.
        """
        assert(len(ind_0) == len(ind_1))

        point = np.random.randint(len(ind_0))
#         new_0, new_1 = np.zeros(len(ind_0)),  np.zeros(len(ind_0))
        new_0 = np.hstack((ind_0[:point], ind_1[point:]))
        new_1 = np.hstack((ind_1[:point], ind_0[point:]))

        assert(len(new_0) == len(ind_0))
        return new_0, new_1

    def mutation(self, indi):
        """
        Simple mutation.
        Arg:
            indi: individual to mutation.
        """
        point = np.random.randint(len(indi))
        indi[point] = 1 - indi[point]
        return indi


    def rws(self, size, fitness):
        """
        Roulette Wheel Selection.
        Args:
            size: the size of population you want to select according to their fitness.
            fitness: the fitness of population you want to apply rws to.
        """
        if self.maximum:
            fitness_ = fitness
        else:
            fitness_ = 1.0 / fitness
#         fitness_ = fitness
        idx = np.random.choice(np.arange(len(fitness_)), size=size, replace=True,
               p=fitness_/fitness_.sum()) 
        return idx

    def run(self):
        """
        Run the genetic algorithm.
        Ret:
            global_best_ind: The best indiviudal during the evolutionary process.
            global_best_fitness: The fitness of the global_best_ind.
        """
        # global_best = 0
        self.initialization()
        best_index = np.argsort(self.fitness)[-1]
        global_best_fitness = self.fitness[best_index]
        global_best_ind = self.pop[best_index, :]
        eva_times = self.pop_shape[0]
        count = 0

        for it in trange(self.max_round):
            next_gene = []

            for n in range(int(self.pop_shape[0]/2)):
                i, j = self.rws(2, self.fitness) # choosing 2 individuals with rws.
                indi_0, indi_1 = self.pop[i, :].copy(), self.pop[j, :].copy()
                if np.random.rand() < self.p_c:
                    indi_0, indi_1 = self.crossover(indi_0, indi_1)

                if np.random.rand() < self.p_m:
                    indi_0 = self.mutation(indi_0)
                    indi_1 = self.mutation(indi_1)

                next_gene.append(indi_0)
                next_gene.append(indi_1)

            self.pop = np.array(next_gene)
            self.fitness = self.evaluation(self.pop)
            eva_times += self.pop_shape[0]

            # Saving the current best genes
            # best_index = np.argsort(self.fitness)[-1]
            # self.generations.append({"gene": self.pop[best_index], "fitness_value":np.max(self.fitness)})

            if self.maximum:
                if np.max(self.fitness) > global_best_fitness:
                    best_index = np.argsort(self.fitness)[-1]
                    global_best_fitness = self.fitness[best_index]
                    global_best_ind = self.pop[best_index, :]
                    count = 0
                else:
                    count +=1
                worst_index = np.argsort(self.fitness)[0]
                self.pop[worst_index, :] = global_best_ind
                self.fitness[worst_index] = global_best_fitness

            else:
                if np.min(self.fitness) < global_best_fitness:
                    best_index = np.argsort(self.fitness)[0]
                    global_best_fitness = self.fitness[best_index]
                    global_best_ind = self.pop[best_index, :]
                    count = 0
                else:
                    count +=1

                worst_index = np.argsort(self.fitness)[-1]
                self.pop[worst_index, :] = global_best_ind
                self.fitness[worst_index] = global_best_fitness

            if self.verbose != None and 0 == (it % self.verbose):
                print('Gene {}:'.format(it))
                print('Global best fitness:', global_best_fitness)

            if self.early_stop_rounds != None and count > self.early_stop_rounds:
                print('Did not improved within {} rounds. Break.'.format(self.early_stop_rounds))
                break

        print('\n Solution: {} \n Fitness: {} \n Evaluation times: {}'.format(global_best_ind, global_best_fitness, eva_times))
        return global_best_ind, global_best_fitness

__author__ = "Vangipuram Srinivasa Sarath Chandra"
__email__ = "vssarathc04@gmail.com"
__version__ = "0.0.1"

if __name__ == '__main__':
    test()