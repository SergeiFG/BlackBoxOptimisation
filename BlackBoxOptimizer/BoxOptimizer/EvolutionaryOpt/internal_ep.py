import numpy as np
from scipy.interpolate import RBFInterpolator

class EvolutionaryProgramming:
    def __init__(self, func, dimension=4, population_size=10, offspring_per_parent=5,
                 mutation_prob=0.3, sigma_init=0.1, t_max=50,
                 lower_bounds=None, upper_bounds=None,
                 output_lower_bounds=None, output_upper_bounds=None):
        self.func = func
        self.dimension = dimension
        self.population_size = population_size
        self.offspring_per_parent = offspring_per_parent
        self.mutation_prob = mutation_prob
        self.sigma_init = sigma_init
        self.t_max = t_max

        # Инициализация границ входных переменных
        self.lower_bounds = np.full(dimension, -np.inf) if lower_bounds is None else np.array(lower_bounds)
        self.upper_bounds = np.full(dimension, np.inf) if upper_bounds is None else np.array(upper_bounds)

        # Инициализация границ выходных переменных (без учета целевой функции)
        self.output_lower_bounds = np.array([]) if output_lower_bounds is None else np.array(output_lower_bounds)
        self.output_upper_bounds = np.array([]) if output_upper_bounds is None else np.array(output_upper_bounds)

        if offspring_per_parent < 1:
            raise ValueError("offspring_per_parent должен быть >= 1")

        self.tau = 1 / np.sqrt(2 * np.sqrt(self.population_size * self.dimension))
        self.tau_prime = 1 / np.sqrt(2 * self.population_size * self.dimension)
        self.min_epsilon = 1e-3
        self.surrogate_update_freq = 5
        self.population = self._initialize_population()
        self.sigmas = np.full((self.population_size, self.dimension), self.sigma_init)
        self.surrogate_model = None
        self.function_values = None

    def _initialize_population(self):
        population = np.zeros((self.population_size, self.dimension))
        for i in range(self.dimension):
            lower = self.lower_bounds[i] if not np.isinf(self.lower_bounds[i]) else -1e10
            upper = self.upper_bounds[i] if not np.isinf(self.upper_bounds[i]) else 1e10
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
        return population

    def _enforce_bounds(self, x):
        x_clipped = np.clip(x, self.lower_bounds, self.upper_bounds)
        for i in range(self.dimension):
            if np.isclose(x_clipped[i], self.lower_bounds[i]):
                x_clipped[i] = self.lower_bounds[i] + 1e-5
            elif np.isclose(x_clipped[i], self.upper_bounds[i]):
                x_clipped[i] = self.upper_bounds[i] - 1e-5
        return x_clipped

    def _check_output_constraints(self, output_values):
        """Проверка ограничений выходных переменных"""
        if len(output_values) <= 1:  # Только целевая функция
            return True
            
        # Проверяем только те параметры, для которых заданы ограничения
        num_output_params = min(len(self.output_lower_bounds), len(output_values)-1)
        
        for i in range(num_output_params):
            if (output_values[i+1] < self.output_lower_bounds[i] or 
                output_values[i+1] > self.output_upper_bounds[i]):
                return False
        return True
                
        for i in range(min(len(output_params), len(self.output_upper_bounds))):
            if (not np.isinf(self.output_upper_bounds[i]) and 
                output_params[i] > self.output_upper_bounds[i]):
                return False
                
        return True

    def _penalize_fitness(self, fitness, output_values):
        """Штрафование fitness при нарушении ограничений выходных переменных"""
        penalty = 0
        output_params = output_values[1:] if len(output_values) > 1 else []
        
        for i in range(min(len(output_params), len(self.output_lower_bounds))):
            if not np.isinf(self.output_lower_bounds[i]):
                penalty += max(self.output_lower_bounds[i] - output_params[i], 0)**2
                
        for i in range(min(len(output_params), len(self.output_upper_bounds))):
            if not np.isinf(self.output_upper_bounds[i]):
                penalty += max(output_params[i] - self.output_upper_bounds[i], 0)**2
                
        return fitness + 1e6 * penalty

    def mutate(self, parent_x, parent_sigma):
        offspring = []
        for _ in range(self.offspring_per_parent):
            child_x = parent_x.copy()
            child_sigma = parent_sigma.copy()
            
            mask = (np.random.rand(self.dimension) <= self.mutation_prob)
            mutation = mask * (child_sigma * np.random.randn(self.dimension))
            
            too_low = (child_x + mutation) <= self.lower_bounds
            too_high = (child_x + mutation) >= self.upper_bounds
            mutation[too_low] *= 0.5
            mutation[too_high] *= 0.5
            
            child_x += mutation
            child_x = self._enforce_bounds(child_x)
            
            child_sigma *= np.exp(self.tau_prime * np.random.randn() + 
                                self.tau * np.random.randn(self.dimension))
            
            output_values = self.func(child_x)
            fitness = output_values[0]  # Первое значение - целевая функция
            
            if not self._check_output_constraints(output_values):
                fitness = self._penalize_fitness(fitness, output_values)
            
            offspring.append({
                'x': child_x,
                'sigma': child_sigma,
                'fitness': fitness,
                'output_values': output_values
            })
        return offspring

    def select_best_offspring(self, offspring):
        if not offspring:
            raise ValueError("No offspring generated! Check mutation parameters.")
        best_offspring = min(offspring, key=lambda ind: ind['fitness'])
        return best_offspring['x'], best_offspring['sigma']

    def run(self):
        for t in range(self.t_max):
            if t == 0:
                all_off = []
                for i in range(self.population_size):
                    off = self.mutate(self.population[i], self.sigmas[i])
                    all_off.extend(off)
                
                self.population = np.array([ind['x'] for ind in all_off])
                self.sigmas = np.array([ind['sigma'] for ind in all_off])
                self.function_values = np.array([ind['fitness'] for ind in all_off])
                self._validate_population()
            else:
                if t % self.surrogate_update_freq == 0:
                    self.update_surrogate()
                
                new_pop = []
                new_sig = []
                new_val = []
                for i in range(self.population_size):
                    off = self.mutate(self.population[i], self.sigmas[i])
                    best = self.select_best_offspring(off)
                    new_pop.append(best[0])
                    new_sig.append(best[1])
                    output = self.func(best[0])
                    fitness = output[0]
                    
                    if not self._check_output_constraints(output):
                        fitness = self._penalize_fitness(fitness, output)
                    
                    new_val.append(fitness)
                
                current_values = np.atleast_1d(self.function_values)
                new_values = np.atleast_1d(np.array(new_val))
                
                comb_pop = np.vstack((self.population, np.array(new_pop)))
                comb_sig = np.vstack((self.sigmas, np.array(new_sig)))
                comb_val = np.hstack((current_values, new_values))
                
                best_idx = np.argsort(comb_val)[:self.population_size]
                self.population = comb_pop[best_idx]
                self.sigmas = comb_sig[best_idx]
                self.function_values = comb_val[best_idx]
                self._validate_population()
        
        i_best = np.argmin(self.function_values)
        return self.population[i_best], self.function_values[i_best]

    def _validate_population(self):
        for i in range(self.population_size):
            if not (np.all(self.population[i] >= self.lower_bounds) and 
                   np.all(self.population[i] <= self.upper_bounds)):
                self.population[i] = self._enforce_bounds(self.population[i])

    def update_surrogate(self):
        epsilon = max(np.mean(np.std(self.population, axis=0)), self.min_epsilon)
        noisy_population = self.population + np.random.normal(0, 1e-8, self.population.shape)
        
        try:
            self.surrogate_model = RBFInterpolator(
                noisy_population,
                self.function_values,
                kernel='linear',
                epsilon=epsilon
            )
        except np.linalg.LinAlgError:
            self.surrogate_model = None



# def test_evolutionary_programming():
#     def test_func(x):
#         # Возвращаем: [целевая функция, выходной параметр 1, выходной параметр 2]
#         return np.array([np.sum(x**2), x[0]*2, x[1]*3])
    
#     # Ограничения входных переменных
#     input_lower = [-2, -3]
#     input_upper = [2, 3]
    
#     # Ограничения выходных переменных (только для выходных параметров)
#     output_lower = [0, -5]  # Для output1 и output2
#     output_upper = [4, 6]   # Для output1 и output2
    
#     ep = EvolutionaryProgramming(
#         func=test_func,
#         dimension=2,
#         population_size=20,
#         lower_bounds=input_lower,
#         upper_bounds=input_upper,
#         output_lower_bounds=output_lower,
#         output_upper_bounds=output_upper
#     )
    
#     best_solution, best_fitness = ep.run()
#     output = test_func(best_solution)
    
#     print("\nРезультаты оптимизации:")
#     print(f"Лучшее решение: {best_solution}")
#     print(f"Значения выходных параметров: {output[1:]}")
#     print(f"Целевая функция: {output[0]}")
    
#     print("\nПроверка ограничений:")
#     # Проверка входных параметров
#     input_check = (np.all(best_solution >= input_lower)) & (np.all(best_solution <= input_upper))
#     print(f"Входные параметры: {'OK' if input_check else 'Нарушено'}")
    
#     # Проверка выходных параметров
#     output_check = True
#     for i, (val, lower, upper) in enumerate(zip(output[1:], output_lower, output_upper)):
#         check = (val >= lower) and (val <= upper)
#         print(f"Выходной параметр {i+1}: {val:.4f} | "
#               f"Границы [{lower}, {upper}] | "
#               f"{'OK' if check else 'Нарушено'}")
#         output_check &= check
    
#     print(f"\nИтоговая проверка: {'Все ограничения соблюдены' if input_check and output_check else 'Есть нарушения'}")

# if __name__ == "__main__":
#     test_evolutionary_programming()