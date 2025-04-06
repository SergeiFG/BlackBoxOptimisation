import numpy as np
from scipy.interpolate import RBFInterpolator

class EvolutionaryProgramming:
    def __init__(self, func, dimension=4, population_size=10, offspring_per_parent=5,
                 mutation_prob=0.3, sigma_init=0.1, t_max=50):
        self.func = func
        self.dimension = dimension
        self.population_size = population_size
        self.offspring_per_parent = offspring_per_parent
        self.mutation_prob = mutation_prob
        self.sigma_init = sigma_init
        self.t_max = t_max

        self.tau = 1 / np.sqrt(2 * np.sqrt(self.population_size * self.dimension))
        self.tau_prime = 1 / np.sqrt(2 * self.population_size * self.dimension)

        self.population = np.random.uniform(-5, 5, (self.population_size, self.dimension)) \
                          + np.random.normal(0, 0.01, (self.population_size, self.dimension))
        self.sigmas = np.full((self.population_size, self.dimension), self.sigma_init)

        self.surrogate_model = None
        self.function_values = None

    def update_surrogate(self):
        epsilon = np.mean(np.std(self.population, axis=0))
        if epsilon < 1e-6:
            epsilon = 1e-3
        try:
            self.surrogate_model = RBFInterpolator(self.population,
                                                    self.function_values,
                                                    kernel='gaussian',
                                                    epsilon=epsilon)
        except np.linalg.LinAlgError:
            # если сингулярность — добавляем шум
            self.population += np.random.normal(0, 0.001, self.population.shape)
            self.surrogate_model = RBFInterpolator(self.population,
                                                    self.function_values,
                                                    kernel='gaussian',
                                                    epsilon=epsilon)

    def mutate(self, parent_x, parent_sigma):
        offspring = []
        for _ in range(self.offspring_per_parent):
            child_x = parent_x.copy()
            child_sigma = parent_sigma.copy()
            mask = (np.random.rand(self.dimension) <= self.mutation_prob)
            child_x += mask * (child_sigma * np.random.randn(self.dimension))
            child_sigma *= np.exp(self.tau_prime * np.random.randn()
                                  + self.tau * np.random.randn(self.dimension))
            offspring.append((child_x, child_sigma))
        return offspring

    def select_best_offspring(self, offspring):
        X = np.array([c[0] for c in offspring])
        preds = self.surrogate_model(X)
        idx = np.argmin(preds)
        return offspring[idx]

    def run(self):
        for t in range(self.t_max):
            if t == 0:
                # первая генерация
                all_off = []
                for i in range(self.population_size):
                    all_off.extend(self.mutate(self.population[i], self.sigmas[i]))
                self.population = np.array([c[0] for c in all_off])
                self.sigmas     = np.array([c[1] for c in all_off])
                self.function_values = np.apply_along_axis(self.func, 1, self.population)
                self.update_surrogate()
            else:
                self.update_surrogate()
                new_pop = []
                new_sig = []
                new_val = []
                for i in range(self.population_size):
                    off = self.mutate(self.population[i], self.sigmas[i])
                    best = self.select_best_offspring(off)
                    true_v = self.func(best[0])
                    new_pop.append(best[0]); new_sig.append(best[1]); new_val.append(true_v)
                # отбор лучших
                comb_pop = np.vstack((self.population, np.array(new_pop)))
                comb_sig = np.vstack((self.sigmas,     np.array(new_sig)))
                comb_val = np.hstack((self.function_values, np.array(new_val)))
                best_idx = np.argsort(comb_val)[:self.population_size]
                self.population = comb_pop[best_idx]
                self.sigmas     = comb_sig[best_idx]
                self.function_values = comb_val[best_idx]
        i_best = np.argmin(self.function_values)
        return self.population[i_best], self.function_values[i_best]