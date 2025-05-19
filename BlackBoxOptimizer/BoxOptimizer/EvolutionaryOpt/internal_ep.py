import numpy as np
from scipy.interpolate import RBFInterpolator

class EvolutionaryProgramming:
    def __init__(self, func, dimension=4, population_size=10, offspring_per_parent=5,
                 mutation_prob=0.3, sigma_init=0.1, t_max=50,
                 lower_bounds=None, upper_bounds=None,
                 output_lower_bounds=None, output_upper_bounds=None,
                 discrete_indices=None):
        """
        Args:
            discrete_indices: list[int] - индексы параметров, которые должны быть 0 или 1
        """
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

        # Инициализация границ выходных переменных
        self.output_lower_bounds = np.array([]) if output_lower_bounds is None else np.array(output_lower_bounds)
        self.output_upper_bounds = np.array([]) if output_upper_bounds is None else np.array(output_upper_bounds)

        if offspring_per_parent < 1:
            raise ValueError("offspring_per_parent должен быть >= 1")

        self.tau = 1 / np.sqrt(2 * np.sqrt(self.population_size * self.dimension))
        self.tau_prime = 1 / np.sqrt(2 * self.population_size * self.dimension)
        self.min_epsilon = 1e-3
        self.surrogate_update_freq = 5

        # Инициализация популяции
        self.population = self._initialize_population()
        self.sigmas = np.full((self.population_size, self.dimension), self.sigma_init)
        self.surrogate_model = None
        self.function_values = None

    def _initialize_population(self):
        """Инициализация популяции с учетом дискретных параметров"""
        population = np.zeros((self.population_size, self.dimension))
        
        for i in range(self.dimension):
            if i in self.discrete_indices:
                # Генерация бинарных значений (0 или 1)
                population[:, i] = np.random.randint(0, 2, size=self.population_size)
            else:
                # Генерация непрерывных значений
                lower = self.lower_bounds[i] if not np.isinf(self.lower_bounds[i]) else -1e10
                upper = self.upper_bounds[i] if not np.isinf(self.upper_bounds[i]) else 1e10
                population[:, i] = np.random.uniform(lower, upper, self.population_size)
                
        return population

    def _enforce_bounds(self, x):
        """Применение ограничений с учетом дискретных параметров"""
        x_clipped = np.clip(x, self.lower_bounds, self.upper_bounds)
        
        # Для дискретных параметров - строго 0 или 1
        for idx in self.discrete_indices:
            x_clipped[idx] = 1 if x_clipped[idx] >= 0.5 else 0
            
        # Для непрерывных параметров - небольшой отступ от границ
        for i in range(self.dimension):
            if i not in self.discrete_indices:
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

    def _penalize_fitness(self, fitness, output_values):
        """Штрафование fitness при нарушении ограничений"""
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
        """Мутация с учетом дискретных параметров"""
        offspring = []
        
        for _ in range(self.offspring_per_parent):
            child_x = parent_x.copy()
            child_sigma = parent_sigma.copy()
            
            # Мутация непрерывных параметров
            continuous_mask = np.ones(self.dimension, dtype=bool)
            if self.discrete_indices:
                continuous_mask[np.array(self.discrete_indices)] = False
                
            mask = (np.random.rand(self.dimension) <= self.mutation_prob) & continuous_mask
            mutation = mask * (child_sigma * np.random.randn(self.dimension))
            
            # Применение мутации
            child_x += mutation
            child_x = self._enforce_bounds(child_x)
            
            # Мутация дискретных параметров (инверсия)
            for idx in self.discrete_indices:
                if np.random.rand() <= self.mutation_prob:
                    child_x[idx] = 1 - child_x[idx]
            
            # Обновление сигм только для непрерывных параметров
            child_sigma[continuous_mask] *= np.exp(
                self.tau_prime * np.random.randn() + 
                self.tau * np.random.randn(np.sum(continuous_mask)))
            
            # Расчет фитнес-функции
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
        """Выбор лучшего потомка"""
        if not offspring:
            raise ValueError("No offspring generated! Check mutation parameters.")
        return min(offspring, key=lambda ind: ind['fitness'])

    def run(self):
        """Основной цикл оптимизации"""
        for t in range(self.t_max):
            if t == 0:
                # Первое поколение
                all_offspring = []
                for i in range(self.population_size):
                    offspring = self.mutate(self.population[i], self.sigmas[i])
                    all_offspring.extend(offspring)
                
                self.population = np.array([ind['x'] for ind in all_offspring])
                self.sigmas = np.array([ind['sigma'] for ind in all_offspring])
                self.function_values = np.array([ind['fitness'] for ind in all_offspring])
            else:
                # Последующие поколения
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
                
                # Объединение старой и новой популяции
                combined_population = np.vstack((self.population, np.array(new_population)))
                combined_sigmas = np.vstack((self.sigmas, np.array(new_sigmas)))
                combined_values = np.hstack((self.function_values, np.array(new_values)))
                
                # Отбор лучших
                best_indices = np.argsort(combined_values)[:self.population_size]
                self.population = combined_population[best_indices]
                self.sigmas = combined_sigmas[best_indices]
                self.function_values = combined_values[best_indices]
        
        # Возвращаем лучшее решение
        best_idx = np.argmin(self.function_values)
        return self.population[best_idx], self.function_values[best_idx]

    def _validate_population(self):
        """Проверка соблюдения ограничений"""
        for i in range(self.population_size):
            self.population[i] = self._enforce_bounds(self.population[i])

    def update_surrogate(self):
        """Обновление суррогатной модели"""
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


def test_evolutionary_programming():
    """Тестовая функция для проверки работы алгоритма"""
    def test_func(x):
        # x[0] - непрерывный параметр
        # x[1] - дискретный параметр (0 или 1)
        # x[2] - дискретный параметр (0 или 1)
        return np.array([np.sum(x**2), x[1], x[2]])
    
    # Параметры оптимизации
    params = {
        'func': test_func,
        'dimension': 3,
        'population_size': 30,
        'offspring_per_parent': 2,
        'mutation_prob': 0.3,
        'sigma_init': 0.2,
        't_max': 100,
        'lower_bounds': [0, 0, 0],
        'upper_bounds': [10, 1, 1],
        'discrete_indices': [1, 2]  # Индексы дискретных параметров
    }
    
    # Создаем оптимизатор
    optimizer = EvolutionaryProgramming(**params)
    
    # Запускаем оптимизацию
    best_solution, best_fitness = optimizer.run()
    output = test_func(best_solution)
    
    # Выводим результаты
    print("\n=== Результаты оптимизации ===")
    print(f"Лучшее решение: {best_solution}")
    print(f"Значение функции: {best_fitness:.6f}")
    print(f"Дискретные параметры: {best_solution[1]} (должен быть 0 или 1), {best_solution[2]} (должен быть 0 или 1)")
    
    # Проверка ограничений
    print("\n=== Проверка ограничений ===")
    print(f"Непрерывный параметр в границах [0, 10]: {0 <= best_solution[0] <= 10}")
    print(f"Дискретные параметры: {best_solution[1] in [0, 1]} и {best_solution[2] in [0, 1]}")


if __name__ == "__main__":
    test_evolutionary_programming()   