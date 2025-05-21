import numpy as np

class GeneticAlgorithmOptimizer:
    def __init__(self, func, dimension=5, population_size=100, generations=150,
                 init_mutation=0.5, min_mutation=0.2, elite_size=5,
                 lower_bounds=None, upper_bounds=None,
                 output_lower_bounds=None, output_upper_bounds=None,
                 discrete_indices=None):
        self.func = func
        self.dimension = dimension
        self.population_size = population_size
        self.generations = generations
        self.init_mutation = init_mutation
        self.min_mutation = min_mutation
        self.elite_size = elite_size
        self.discrete_indices = discrete_indices if discrete_indices is not None else []

        # Установка границ входных параметров
        self.lower_bounds = np.full(dimension, -np.inf) if lower_bounds is None else np.array(lower_bounds)
        self.upper_bounds = np.full(dimension, np.inf) if upper_bounds is None else np.array(upper_bounds)
        
        # Автоматические границы для булевых параметров
        for idx in self.discrete_indices:
            self.lower_bounds[idx] = 0
            self.upper_bounds[idx] = 1

        # Границы выходных параметров
        self.output_lower_bounds = np.array([]) if output_lower_bounds is None else np.array(output_lower_bounds)
        self.output_upper_bounds = np.array([]) if output_upper_bounds is None else np.array(output_upper_bounds)
        
        # История
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.mutation_rates = []
        self.output_values_history = []

    def _initialize_population(self):
        """Инициализация популяции с булевыми параметрами"""
        population = np.empty((self.population_size, self.dimension))
        for i in range(self.dimension):
            if i in self.discrete_indices:
                population[:, i] = np.random.randint(0, 2, self.population_size)
            else:
                population[:, i] = np.random.uniform(
                    self.lower_bounds[i],
                    self.upper_bounds[i],
                    self.population_size
                )
        return population

    def _get_current_mutation_rate(self, generation):
        """Динамическая вероятность мутации"""
        decay_rate = np.log(self.init_mutation / self.min_mutation) / self.generations
        return self.init_mutation * np.exp(-decay_rate * generation)

    def _check_output_constraints(self, output_values):
        """Проверка ограничений выходных параметров"""
        if len(output_values) <= 1:
            return True
        num_output_params = min(len(self.output_lower_bounds), len(output_values)-1)
        for i in range(num_output_params):
            if (output_values[i+1] < self.output_lower_bounds[i] or 
                output_values[i+1] > self.output_upper_bounds[i]):
                return False
        return True

    def _penalize_fitness(self, fitness, output_values):
        """Штраф за нарушение ограничений"""
        penalty = 0
        output_params = output_values[1:] if len(output_values) > 1 else []
        for i in range(min(len(output_params), len(self.output_lower_bounds))):
            penalty += max(self.output_lower_bounds[i] - output_params[i], 0)**2
        for i in range(min(len(output_params), len(self.output_upper_bounds))):
            penalty += max(output_params[i] - self.output_upper_bounds[i], 0)**2
        return fitness + 1e6 * penalty

    def _evaluate(self, population, ask_time=100):
        """Оценка популяции"""
        fitness_values = []
        output_values_list = []
        for ind in population:
            output_values = self.func(ind)
            fitness = output_values[0]
            if not self._check_output_constraints(output_values):
                fitness = self._penalize_fitness(fitness, output_values)
            fitness_values.append(fitness)
            output_values_list.append(output_values)
        return np.array(fitness_values), np.array(output_values_list)

    def _selection(self, population, fitness, num_parents):
        """Рулеточный отбор"""
        inverted_fitness = 1 / (1 + fitness)
        probabilities = inverted_fitness / np.sum(inverted_fitness)
        parent_indices = np.random.choice(range(len(population)), size=num_parents, p=probabilities)
        return population[parent_indices]

    def _crossover(self, parents, offspring_size):
        """Кроссовер с разными операторами"""
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0]):
            p1, p2 = np.random.choice(parents.shape[0], 2, replace=False)
            crossover_type = np.random.choice(['arithmetic', 'blend', 'single_point'], p=[0.5, 0.3, 0.2])
            
            if crossover_type == 'arithmetic':
                alpha = np.random.random()
                offspring[k] = alpha * parents[p1] + (1 - alpha) * parents[p2]
            elif crossover_type == 'blend':
                for i in range(self.dimension):
                    x1, x2 = sorted([parents[p1][i], parents[p2][i]])
                    diff = x2 - x1
                    low = max(x1 - 0.5 * diff, self.lower_bounds[i])
                    high = min(x2 + 0.5 * diff, self.upper_bounds[i])
                    offspring[k][i] = np.random.uniform(low, high)
            else:
                point = np.random.randint(1, self.dimension)
                offspring[k][:point] = parents[p1][:point]
                offspring[k][point:] = parents[p2][point:]
                
            offspring[k] = np.clip(offspring[k], self.lower_bounds, self.upper_bounds)
        return offspring

    def _mutate(self, offspring, generation):
        """Мутация с битовой инверсией для булевых параметров"""
        current_mutation_rate = self._get_current_mutation_rate(generation)
        mutation_strength = 0.5 * (1 - generation / self.generations)
        
        for i in range(offspring.shape[0]):
            if np.random.random() < current_mutation_rate:
                for j in range(self.dimension):
                    if j in self.discrete_indices:
                        offspring[i][j] = 1 - offspring[i][j]
                    else:
                        mutation = np.random.normal(0, mutation_strength)
                        offspring[i][j] = np.clip(
                            offspring[i][j] + mutation,
                            self.lower_bounds[j],
                            self.upper_bounds[j]
                        )
        return offspring

    def _enforce_bounds(self, x):
        """Коррекция границ для булевых параметров"""
        x_clipped = np.clip(x, self.lower_bounds, self.upper_bounds)
        for idx in self.discrete_indices:
            x_clipped[idx] = 1 if x_clipped[idx] >= 0.5 else 0
        return x_clipped

    def run(self):
        """Запуск алгоритма"""
        population = self._initialize_population()
        for gen in range(self.generations):
            fitness, output_values = self._evaluate(population)
            self.mutation_rates.append(self._get_current_mutation_rate(gen))
            self.best_fitness_history.append(np.min(fitness))
            self.avg_fitness_history.append(np.mean(fitness))
            
            # Элитизм
            elite_indices = np.argsort(fitness)[:self.elite_size]
            elite = population[elite_indices]
            
            # Селекция и кроссовер
            parents = self._selection(population, fitness, self.population_size - self.elite_size)
            offspring = self._crossover(parents, (self.population_size - self.elite_size, self.dimension))
            
            # Мутация
            mutated_offspring = self._mutate(offspring, gen)
            population = np.vstack([mutated_offspring, elite])
            population = np.array([self._enforce_bounds(ind) for ind in population])

        # Финал
        fitness, output_values = self._evaluate(population)
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx]
        self.best_fitness = fitness[best_idx]
        self.best_output_values = output_values[best_idx]
        return self.best_solution, self.best_fitness, self.best_output_values