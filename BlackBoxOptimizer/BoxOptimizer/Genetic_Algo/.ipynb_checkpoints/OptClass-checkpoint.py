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

        # Инициализация границ
        self.lower_bounds = np.full(dimension, -np.inf)
        self.upper_bounds = np.full(dimension, np.inf)

        # Применение пользовательских границ
        if lower_bounds is not None:
            self.lower_bounds[:len(lower_bounds)] = lower_bounds
        if upper_bounds is not None:
            self.upper_bounds[:len(upper_bounds)] = upper_bounds

        # Установка [0,1] для дискретных параметров
        for idx in self.discrete_indices:
            self.lower_bounds[idx] = 0
            self.upper_bounds[idx] = 1

        # Вычисление максимального конечного диапазона
        self.max_finite_diap = self._calculate_max_finite_diap()

        # Корректировка границ для непрерывных параметров
        for j in range(self.dimension):
            if j in self.discrete_indices:
                continue
            lb = self.lower_bounds[j]
            ub = self.upper_bounds[j]
            if np.isfinite(lb) and np.isfinite(ub):
                continue
            elif np.isfinite(ub):
                self.lower_bounds[j] = ub - self.max_finite_diap
            elif np.isfinite(lb):
                self.upper_bounds[j] = lb + self.max_finite_diap
            else:
                self.lower_bounds[j] = -self.max_finite_diap / 2
                self.upper_bounds[j] = self.max_finite_diap / 2

        # Инициализация остальных параметров
        self.output_lower_bounds = np.array([]) if output_lower_bounds is None else np.array(output_lower_bounds)
        self.output_upper_bounds = np.array([]) if output_upper_bounds is None else np.array(output_upper_bounds)
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.mutation_rates = []
        self.output_values_history = []
        self.best_individuals_history = []
        self.num_valid_solutions_history = []

    def _calculate_max_finite_diap(self):
        max_diap = -np.inf
        for j in range(self.dimension):
            lb = self.lower_bounds[j]
            ub = self.upper_bounds[j]
            if np.isfinite(lb) and np.isfinite(ub) and j not in self.discrete_indices:
                current_diap = ub - lb
                if current_diap > max_diap:
                    max_diap = current_diap
        return max_diap if np.isfinite(max_diap) else 1000.0
    

    def _initialize_population(self):
        population = np.empty((self.population_size, self.dimension))
        for i in range(self.dimension):
            if i in self.discrete_indices:
                population[:, i] = np.random.randint(0, 2, self.population_size)
            else:
                population[:, i] = 10*np.random.uniform(
                    self.lower_bounds[i],
                    self.upper_bounds[i],
                    self.population_size
                )
        return population

    def _get_current_mutation_rate(self, generation):
        decay_rate = np.log(self.init_mutation / self.min_mutation) / self.generations
        return self.init_mutation * np.exp(-decay_rate * generation)

    def _check_output_constraints(self, output_values):
        if len(self.output_lower_bounds) == 0 or len(self.output_upper_bounds) == 0:
            return True
            
        if len(output_values) <= 1:
            return True
            
        num_output_params = min(len(self.output_lower_bounds), len(output_values)-1)
        for i in range(num_output_params):
            if (output_values[i+1] < self.output_lower_bounds[i] or 
                output_values[i+1] > self.output_upper_bounds[i]):
                return False
        return True

    def _evaluate(self, population, ask_time=100):
        fitness_values = []
        output_values_list = []
        valid_indices = []
        
        for idx, ind in enumerate(population):
            output_values = self.func(ind)
            if self._check_output_constraints(output_values):
                fitness_values.append(output_values[0])
                output_values_list.append(output_values)
                valid_indices.append(idx)
        
        # Если нет ни одного допустимого решения, возвращаем худшие возможные значения
        if not fitness_values:
            return np.full(len(population), np.inf), np.array([])
            
        # Для недопустимых решений устанавливаем худшую возможную приспособленность
        final_fitness = np.full(len(population), np.inf)
        final_outputs = np.array([None] * len(population))
        
        for i, idx in enumerate(valid_indices):
            final_fitness[idx] = fitness_values[i]
            final_outputs[idx] = output_values_list[i]
            
        return final_fitness, final_outputs

    def _selection(self, population, fitness, num_parents):
        # Выбираем только допустимые решения (с конечной приспособленностью)
        valid_indices = np.where(np.isfinite(fitness))[0]
        if len(valid_indices) == 0:
            return population[np.random.choice(len(population), size=num_parents, replace=True)]
            
        valid_population = population[valid_indices]
        valid_fitness = fitness[valid_indices]
        
        inverted_fitness = 1 / (1 + valid_fitness)
        probabilities = inverted_fitness / np.sum(inverted_fitness)
        parent_indices = np.random.choice(range(len(valid_population)), size=num_parents, p=probabilities)
        return valid_population[parent_indices]

    def _crossover(self, parents, offspring_size):
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
        current_mutation_rate = self._get_current_mutation_rate(generation)
        mutation_strength = 0.5 * (1 - generation / self.generations)
        
        for i in range(offspring.shape[0]):
            if np.random.random() < current_mutation_rate:
                for j in range(self.dimension):
                    if j in self.discrete_indices:
                        offspring[i][j] = 1 - offspring[i][j]
                    else:
                        lb = self.lower_bounds[j]
                        ub = self.upper_bounds[j]
                        diap = ub - lb if np.isfinite(lb) and np.isfinite(ub) else self.max_finite_diap
                        max_step = diap * mutation_strength
                        mutation = np.random.uniform(-max_step, max_step)
                        new_value = offspring[i][j] + mutation
                        new_value = np.clip(new_value, lb, ub) if np.isfinite(lb) and np.isfinite(ub) else max(lb, min(ub, new_value))
                        offspring[i][j] = new_value
        return offspring

    def _enforce_bounds(self, x):
        x_clipped = np.clip(x, self.lower_bounds, self.upper_bounds)
        for idx in self.discrete_indices:
            x_clipped[idx] = 1 if x_clipped[idx] >= 0.5 else 0
        return x_clipped

    def run(self):
        population = self._initialize_population()
        for gen in range(self.generations):
            fitness, output_values = self._evaluate(population)
            self.mutation_rates.append(self._get_current_mutation_rate(gen))

            # Считаем количество допустимых решений
            valid_count = np.sum(np.isfinite(fitness))
            self.num_valid_solutions_history.append(valid_count)  # Записываем в историю
            
            # Находим лучшее допустимое решение
            valid_indices = np.where(np.isfinite(fitness))[0]
            if len(valid_indices) > 0:
                best_valid_idx = valid_indices[np.argmin(fitness[valid_indices])]
                self.best_fitness_history.append(fitness[best_valid_idx])
                self.best_individuals_history.append(population[best_valid_idx].copy())
            else:
                self.best_fitness_history.append(np.inf)
                self.best_individuals_history.append(population[0].copy())
                
            self.avg_fitness_history.append(np.mean(fitness[np.isfinite(fitness)]) if np.any(np.isfinite(fitness)) else np.inf)
            
            # Элитизм - выбираем только допустимые решения
            valid_indices = np.where(np.isfinite(fitness))[0]
            if len(valid_indices) > 0:
                valid_fitness = fitness[valid_indices]
                valid_population = population[valid_indices]
                elite_indices = np.argsort(valid_fitness)[:min(self.elite_size, len(valid_fitness))]
                elite = valid_population[elite_indices]
            else:
                elite = population[:self.elite_size]  # если нет допустимых, берем первых
                
            # Селекция и кроссовер
            parents = self._selection(population, fitness, self.population_size - len(elite))
            offspring = self._crossover(parents, (self.population_size - len(elite), self.dimension))
            
            # Мутация
            mutated_offspring = self._mutate(offspring, gen)
            population = np.vstack([mutated_offspring, elite])
            population = np.array([self._enforce_bounds(ind) for ind in population])

        # Финал
        fitness, output_values = self._evaluate(population)
        valid_indices = np.where(np.isfinite(fitness))[0]
        if len(valid_indices) > 0:
            best_valid_idx = valid_indices[np.argmin(fitness[valid_indices])]
            self.best_solution = population[best_valid_idx]
            self.best_fitness = fitness[best_valid_idx]
            self.best_output_values = output_values[best_valid_idx]
        else:
            self.best_solution = population[0]
            self.best_fitness = np.inf
            self.best_output_values = None
            
        return self.best_solution, self.best_fitness, self.best_output_values