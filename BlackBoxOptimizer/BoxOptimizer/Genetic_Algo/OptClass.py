import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class GeneticAlgorithmOptimizer:
    def __init__(self, func, dimension=5, population_size=100, generations=150,
                 init_mutation=0.5, min_mutation=0.2, elite_size=5,
                 lower_bounds=None, upper_bounds=None,
                 output_lower_bounds=None, output_upper_bounds=None,
                 discrete_indices=None, base_penalty=1e6, adaptive_penalty=True,
                 penalty_exponent=2, feasibility_phase_generations=0.3):
        self.func = func
        self.dimension = dimension
        self.population_size = population_size
        self.generations = generations
        self.init_mutation = init_mutation
        self.min_mutation = min_mutation
        self.elite_size = elite_size
        self.discrete_indices = discrete_indices if discrete_indices is not None else []
        self.base_penalty = base_penalty
        self.adaptive_penalty = adaptive_penalty
        self.penalty_exponent = penalty_exponent
        self.current_generation = 0
        self.feasibility_phase = True
        self.feasibility_phase_generations = int(generations * feasibility_phase_generations)
        self.best_feasible_solution = None
        self.best_feasible_fitness = np.inf
        self.best_feasible_output = None
        self.best_infeasible_solution = None
        self.best_infeasible_violation = np.inf
        self.best_infeasible_output = None

        # Инициализация границ входных параметров
        self.lower_bounds = np.full(dimension, -np.inf)
        self.upper_bounds = np.full(dimension, np.inf)

        if lower_bounds is not None:
            self.lower_bounds[:len(lower_bounds)] = lower_bounds[:dimension]
        if upper_bounds is not None:
            self.upper_bounds[:len(upper_bounds)] = upper_bounds[:dimension]

        for idx in self.discrete_indices:
            if idx < dimension:
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

        # Инициализация границ выходных параметров (включая целевую функцию)
        self.output_lower_bounds = np.array([])
        self.output_upper_bounds = np.array([])
        
        if output_lower_bounds is not None:
            self.output_lower_bounds = np.array(output_lower_bounds).flatten()
        if output_upper_bounds is not None:
            self.output_upper_bounds = np.array(output_upper_bounds).flatten()
            
        # Автоматическое расширение границ если размеры не совпадают
        n_output = max(len(self.output_lower_bounds), len(self.output_upper_bounds))
        if n_output > 0:
            if len(self.output_lower_bounds) == 0:
                self.output_lower_bounds = np.full(n_output, -np.inf)
            elif len(self.output_lower_bounds) < n_output:
                self.output_lower_bounds = np.pad(
                    self.output_lower_bounds, 
                    (0, n_output - len(self.output_lower_bounds)),
                    'constant',
                    constant_values=-np.inf
                )
                
            if len(self.output_upper_bounds) == 0:
                self.output_upper_bounds = np.full(n_output, np.inf)
            elif len(self.output_upper_bounds) < n_output:
                self.output_upper_bounds = np.pad(
                    self.output_upper_bounds, 
                    (0, n_output - len(self.output_upper_bounds)),
                    'constant',
                    constant_values=np.inf
                )

        # Исторические данные (только необходимые)
        self.best_individuals_history = []  # Лучший вектор MV за поколение
        self.output_values_history = []     # Соответствующий вектор CV
        self.num_valid_solutions_history = []  # Количество допустимых решений
        self.avg_fitness_history = []       # Среднее значение целевой функции

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
                population[:, i] = np.random.uniform(
                    self.lower_bounds[i],
                    self.upper_bounds[i],
                    self.population_size
                )
        return population

    def _get_current_mutation_rate(self, generation):
        decay_rate = np.log(self.init_mutation / self.min_mutation) / self.generations
        return self.init_mutation * np.exp(-decay_rate * generation)

    def _calculate_constraint_violation(self, output_values):
        """Вычисляет сумму нарушений по всем выходным параметрам (включая целевую функцию)"""
        if len(self.output_lower_bounds) == 0 or len(output_values) == 0:
            return 0.0
            
        # Проверяем все выходные параметры
        n = min(len(output_values), len(self.output_lower_bounds))
        
        if n == 0:
            return 0.0
            
        # Вычисляем нарушения для каждой границы
        lower_violations = np.maximum(self.output_lower_bounds[:n] - output_values[:n], 0)
        upper_violations = np.maximum(output_values[:n] - self.output_upper_bounds[:n], 0)
        
        # Суммируем квадраты нарушений
        total_violation = np.sum(lower_violations**2) + np.sum(upper_violations**2)
        return total_violation

    def _get_penalty_coeff(self):
        if not self.adaptive_penalty:
            return self.base_penalty
            
        progress = min(1.0, max(0.0, self.current_generation / self.generations))
        return self.base_penalty * (progress ** self.penalty_exponent)

    def _is_feasible(self, output_values):
        """Проверяет, удовлетворяет ли решение всем ограничениям (включая целевую функцию)"""
        if len(self.output_lower_bounds) == 0 or len(output_values) == 0:
            return True
            
        n = min(len(output_values), len(self.output_lower_bounds))
        
        for i in range(n):
            if (output_values[i] < self.output_lower_bounds[i] or 
                output_values[i] > self.output_upper_bounds[i]):
                return False
        return True

    def _evaluate(self, population):
        fitness_values = []
        output_values_list = []
        violations = []
        feasibility = []
        
        for ind in population:
            output = self.func(ind)
            violation = self._calculate_constraint_violation(output)
            is_feasible = self._is_feasible(output)
            
            violations.append(violation)
            feasibility.append(is_feasible)
            output_values_list.append(output)
            
            # Двухэтапная оценка приспособленности
            if self.feasibility_phase:
                # В фазе поиска допустимых решений минимизируем нарушения
                fitness = violation
            else:
                # В фазе оптимизации:
                if is_feasible:
                    # Для допустимых решений используем целевую функцию (первый выход)
                    fitness = output[0]
                else:
                    # Для недопустимых - большой штраф
                    fitness = 1e20 + violation
                    
            fitness_values.append(fitness)
        
        return (
            np.array(fitness_values),
            np.array(output_values_list, dtype=object),
            np.array(violations),
            np.array(feasibility)
        )

    def _selection(self, population, fitness, num_parents):
        # Все решения имеют конечную приспособленность
        # Нормализуем приспособленность для избежания проблем с нулями
        min_fitness = np.min(fitness)
        normalized_fitness = fitness - min_fitness + 1e-10
        
        # Инвертируем приспособленность для минимизации
        inverted_fitness = 1 / normalized_fitness
        probabilities = inverted_fitness / np.sum(inverted_fitness)
        
        parent_indices = np.random.choice(
            range(len(population)), 
            size=num_parents, 
            p=probabilities,
            replace=True
        )
        return population[parent_indices]

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
                        if np.isfinite(lb) and np.isfinite(ub):
                            new_value = np.clip(new_value, lb, ub)
                        else:
                            if np.isfinite(lb): new_value = max(lb, new_value)
                            if np.isfinite(ub): new_value = min(ub, new_value)
                        offspring[i][j] = new_value
        return offspring

    def _enforce_bounds(self, x):
        x_clipped = np.clip(x, self.lower_bounds, self.upper_bounds)
        for idx in self.discrete_indices:
            if idx < len(x_clipped):
                x_clipped[idx] = 1 if x_clipped[idx] >= 0.5 else 0
        return x_clipped

    def run(self):
        population = self._initialize_population()
        
        for gen in range(self.generations):
            self.current_generation = gen
            
            # Определяем текущую фазу
            if gen > self.feasibility_phase_generations:
                self.feasibility_phase = False
                
            # Оценка популяции
            fitness, outputs, violations, feasibility = self._evaluate(population)
            feasible_indices = np.where(feasibility)[0]
            
            # Сохраняем лучшее допустимое решение (по CV0)
            if len(feasible_indices) > 0:
                feasible_cv0 = [outputs[i][0] for i in feasible_indices]
                best_feasible_idx_in_feasible = np.argmin(feasible_cv0)
                best_feasible_idx = feasible_indices[best_feasible_idx_in_feasible]
                candidate_cv0 = feasible_cv0[best_feasible_idx_in_feasible]
                
                if candidate_cv0 < self.best_feasible_fitness:
                    self.best_feasible_solution = population[best_feasible_idx].copy()
                    self.best_feasible_fitness = candidate_cv0
                    self.best_feasible_output = outputs[best_feasible_idx]
            
            # Сохраняем лучшее недопустимое решение (по минимальному нарушению)
            for i in range(len(population)):
                if not feasibility[i]:
                    if violations[i] < self.best_infeasible_violation:
                        self.best_infeasible_solution = population[i].copy()
                        self.best_infeasible_violation = violations[i]
                        self.best_infeasible_output = outputs[i]
            
            # Сохраняем историю текущего поколения
            all_cv0 = [output[0] for output in outputs]
            best_idx = np.argmin(fitness)
            
            self.best_individuals_history.append(population[best_idx].copy())
            self.output_values_history.append(outputs[best_idx])
            self.num_valid_solutions_history.append(len(feasible_indices))
            self.avg_fitness_history.append(np.mean(all_cv0))
            
            # Элитизм - лучшие решения в текущей фазе
            elite_indices = np.argsort(fitness)[:self.elite_size]
            elite = population[elite_indices]
            
            # Селекция и кроссовер
            parents = self._selection(population, fitness, self.population_size - len(elite))
            offspring = self._crossover(parents, (self.population_size - len(elite), self.dimension))
            
            # Мутация
            mutated_offspring = self._mutate(offspring, gen)
            
            # Формирование новой популяции
            population = np.vstack([mutated_offspring, elite])
            population = np.array([self._enforce_bounds(ind) for ind in population])

        # После завершения всех поколений:
        # Если нашли допустимые решения - возвращаем лучшее
        if self.best_feasible_solution is not None:
            return (
                self.best_feasible_solution, 
                self.best_feasible_fitness, 
                self.best_feasible_output
            )
        
        # Если не нашли допустимых решений - возвращаем лучшее недопустимое
        return (
            self.best_infeasible_solution, 
            self.best_infeasible_output[0], 
            self.best_infeasible_output
        )