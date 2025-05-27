import numpy as np
from scipy.interpolate import RBFInterpolator

class EvolutionaryProgramming:
    def __init__(self, func, dimension=4, population_size=10, offspring_per_parent=5,
                 mutation_prob=0.3, sigma_init=0.1, t_max=50,
                 lower_bounds=None, upper_bounds=None,
                 output_lower_bounds=None, output_upper_bounds=None,
                 discrete_indices=None):
        self.func = func
        self.dimension = dimension
        self.population_size = population_size
        self.offspring_per_parent = offspring_per_parent
        self.mutation_prob = mutation_prob
        self.sigma_init = sigma_init
        self.t_max = t_max
        self.discrete_indices = discrete_indices if discrete_indices is not None else []
        
        # Проверка корректности дискретных индексов
        for idx in self.discrete_indices:
            if idx >= dimension:
                raise ValueError(f"Индекс дискретного параметра {idx} превышает размерность {dimension}")

        # Инициализация границ
        self.lower_bounds = np.full(dimension, -np.inf) if lower_bounds is None else np.array(lower_bounds)
        self.upper_bounds = np.full(dimension, np.inf) if upper_bounds is None else np.array(upper_bounds)
        self.output_lower_bounds = np.array([]) if output_lower_bounds is None else np.array(output_lower_bounds)
        self.output_upper_bounds = np.array([]) if output_upper_bounds is None else np.array(output_upper_bounds)

        if offspring_per_parent < 1:
            raise ValueError("offspring_per_parent должен быть >= 1")

        self.tau = 1 / np.sqrt(2 * np.sqrt(self.population_size * self.dimension))
        self.tau_prime = 1 / np.sqrt(2 * self.population_size * self.dimension)
        self.min_epsilon = 1e-3
        self.surrogate_update_freq = 5
        self.history = []  # Для хранения истории оптимизации

        # Инициализация популяции
        self.population = self._initialize_population()
        self.sigmas = np.full((self.population_size, self.dimension), self.sigma_init)
        self.surrogate_model = None
        self.function_values = None

    def _initialize_population(self):
        population = np.zeros((self.population_size, self.dimension))
        for i in range(self.dimension):
            if i in self.discrete_indices:
                population[:, i] = np.random.randint(0, 1, size=self.population_size)
            else:
                lower = self.lower_bounds[i] if not np.isinf(self.lower_bounds[i]) else -1e10
                upper = self.upper_bounds[i] if not np.isinf(self.upper_bounds[i]) else 1e10
                population[:, i] = 10*np.random.uniform(lower, upper, self.population_size)
        return population

    def _enforce_bounds(self, x):
        x_clipped = np.clip(x, self.lower_bounds, self.upper_bounds)
        for idx in self.discrete_indices:
            x_clipped[idx] = 1 if x_clipped[idx] >= 0.5 else 0
        for i in range(self.dimension):
            if i not in self.discrete_indices:
                if np.isclose(x_clipped[i], self.lower_bounds[i]):
                    x_clipped[i] = self.lower_bounds[i] + 1e-5
                elif np.isclose(x_clipped[i], self.upper_bounds[i]):
                    x_clipped[i] = self.upper_bounds[i] - 1e-5
        return x_clipped

    def _check_output_constraints(self, output_values):
        if len(output_values) <= 1:
            return True
        num_output_params = min(len(self.output_lower_bounds), len(output_values)-1)
        for i in range(num_output_params):
            if (output_values[i+1] < self.output_lower_bounds[i] or 
                output_values[i+1] > self.output_upper_bounds[i]):
                return False
        return True

    def _penalize_fitness(self, fitness, output_values):
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
            
            continuous_mask = np.ones(self.dimension, dtype=bool)
            if self.discrete_indices:
                continuous_mask[np.array(self.discrete_indices)] = False
                
            mask = (np.random.rand(self.dimension) <= self.mutation_prob) & continuous_mask
            mutation = mask * (child_sigma * np.random.randn(self.dimension))
            
            child_x += mutation
            child_x = self._enforce_bounds(child_x)
            
            for idx in self.discrete_indices:
                if np.random.rand() <= self.mutation_prob:
                    child_x[idx] = 1 - child_x[idx]
            
            child_sigma[continuous_mask] *= np.exp(
                self.tau_prime * np.random.randn() + 
                self.tau * np.random.randn(np.sum(continuous_mask)))
            
            output_values = self.func(child_x)
            fitness = output_values[0]
            
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
        return min(offspring, key=lambda ind: ind['fitness'])

    def run(self):
        self.history = []  # Сбрасываем историю перед запуском
        for t in range(self.t_max):
            if t == 0:
                all_offspring = []
                for i in range(self.population_size):
                    offspring = self.mutate(self.population[i], self.sigmas[i])
                    all_offspring.extend(offspring)
                
                self.population = np.array([ind['x'] for ind in all_offspring])
                self.sigmas = np.array([ind['sigma'] for ind in all_offspring])
                self.function_values = np.array([ind['fitness'] for ind in all_offspring])
            else:
                new_population = []
                new_sigmas = []
                new_values = []
                
                for i in range(self.population_size):
                    offspring = self.mutate(self.population[i], self.sigmas[i])
                    best = self.select_best_offspring(offspring)
                    new_population.append(best['x'])
                    new_sigmas.append(best['sigma'])
                    output = self.func(best['x'])
                    fitness = output[0]
                    
                    if not self._check_output_constraints(output):
                        fitness = self._penalize_fitness(fitness, output)
                    
                    new_values.append(fitness)
                
                combined_population = np.vstack((self.population, np.array(new_population)))
                combined_sigmas = np.vstack((self.sigmas, np.array(new_sigmas)))
                combined_values = np.hstack((self.function_values, np.array(new_values)))
                
                best_indices = np.argsort(combined_values)[:self.population_size]
                self.population = combined_population[best_indices]
                self.sigmas = combined_sigmas[best_indices]
                self.function_values = combined_values[best_indices]
            
            # Сохраняем информацию о текущем поколении
            current_best_idx = np.argmin(self.function_values)
            self.history.append({
                'generation': t,
                'best_solution': self.population[current_best_idx].copy(),
                'best_fitness': self.function_values[current_best_idx],
                'average_fitness': np.mean(self.function_values),
                'sigma_values': self.sigmas[current_best_idx].copy(),
                'population_diversity': np.mean(np.std(self.population, axis=0))
            })
        
        best_idx = np.argmin(self.function_values)
        return self.population[best_idx], self.function_values[best_idx], self.history

    def get_history(self):
        return self.history

    def _validate_population(self):
        for i in range(self.population_size):
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