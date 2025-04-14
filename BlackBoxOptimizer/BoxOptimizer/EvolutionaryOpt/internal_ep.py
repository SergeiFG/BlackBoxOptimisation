import numpy as np
from scipy.interpolate import RBFInterpolator

class EvolutionaryProgramming:
    def __init__(self, func, dimension=4, population_size=10, offspring_per_parent=5,
                 mutation_prob=0.3, sigma_init=0.1, t_max=50,
                 lower_bounds=None, upper_bounds=None):
        self.func = func
        self.dimension = dimension
        self.population_size = population_size
        self.offspring_per_parent = offspring_per_parent
        self.mutation_prob = mutation_prob
        self.sigma_init = sigma_init
        self.t_max = t_max

        # Инициализация границ
        self.lower_bounds = np.full(dimension, -np.inf) if lower_bounds is None else np.array(lower_bounds)
        self.upper_bounds = np.full(dimension, np.inf) if upper_bounds is None else np.array(upper_bounds)

        if offspring_per_parent < 1:
            raise ValueError("offspring_per_parent должен быть >= 1")

        self.tau = 1 / np.sqrt(2 * np.sqrt(self.population_size * self.dimension))
        self.tau_prime = 1 / np.sqrt(2 * self.population_size * self.dimension)
        self.min_epsilon = 1e-3  # Минимальное значение для epsilon
        self.surrogate_update_freq = 5  # Частота обновления суррогатной модели

        # Инициализация популяции с гарантированным соблюдением границ
        self.population = self._initialize_population()
        self.sigmas = np.full((self.population_size, self.dimension), self.sigma_init)
        self.surrogate_model = None
        self.function_values = None

    def _initialize_population(self):
        """Генерация начальной популяции с соблюдением границ"""
        population = np.zeros((self.population_size, self.dimension))
        for i in range(self.dimension):
            population[:, i] = np.random.uniform(
                max(self.lower_bounds[i], -5),  # Нижняя граница
                min(self.upper_bounds[i], 5),   # Верхняя граница
                self.population_size
            )
        return population

    def _enforce_bounds(self, x):
        """Строгое соблюдение границ с защитой от численных погрешностей"""
        x_clipped = np.clip(x, self.lower_bounds, self.upper_bounds)
        # Дополнительная проверка для каждого измерения
        for i in range(self.dimension):
            if np.isclose(x_clipped[i], self.lower_bounds[i]):
                x_clipped[i] = self.lower_bounds[i] + 1e-5
            elif np.isclose(x_clipped[i], self.upper_bounds[i]):
                x_clipped[i] = self.upper_bounds[i] - 1e-5
        return x_clipped

    def mutate(self, parent_x, parent_sigma):
        offspring = []
        for _ in range(self.offspring_per_parent):
            child_x = parent_x.copy()
            child_sigma = parent_sigma.copy()
            
            # Применяем мутацию с контролем у границ
            mask = (np.random.rand(self.dimension) <= self.mutation_prob)
            mutation = mask * (child_sigma * np.random.randn(self.dimension))
            
            # Плавное ограничение у границ
            too_low = (child_x + mutation) <= self.lower_bounds
            too_high = (child_x + mutation) >= self.upper_bounds
            mutation[too_low] *= 0.5
            mutation[too_high] *= 0.5
            
            child_x += mutation
            child_x = self._enforce_bounds(child_x)
            
            # Мутация параметров мутации
            child_sigma *= np.exp(self.tau_prime * np.random.randn() + 
                                self.tau * np.random.randn(self.dimension))
            
            # Вычисление fitness
            fitness = self.func(child_x)
            offspring.append({
                'x': child_x,
                'sigma': child_sigma,
                'fitness': fitness
            })
        return offspring

    def update_surrogate(self):
        """Обновление суррогатной модели с защитой от сингулярности"""
        epsilon = max(np.mean(np.std(self.population, axis=0)), self.min_epsilon)
        
        # Добавляем небольшой шум для предотвращения сингулярности
        noisy_population = self.population + np.random.normal(0, 1e-8, self.population.shape)
        
        try:
            self.surrogate_model = RBFInterpolator(
                noisy_population,
                self.function_values,
                kernel='linear',  # Используем линейное ядро для стабильности
                epsilon=epsilon
            )
        except np.linalg.LinAlgError:
            # Fallback: отключаем суррогатную модель при ошибке
            self.surrogate_model = None

    def select_best_offspring(self, offspring):
        if not offspring:
            raise ValueError("No offspring generated! Check mutation parameters.")
        best_offspring = min(offspring, key=lambda ind: ind['fitness'])
        return best_offspring['x'], best_offspring['sigma']

    def run(self):
        for t in range(self.t_max):
            if t == 0:
                # Первая генерация
                all_off = []
                for i in range(self.population_size):
                    off = self.mutate(self.population[i], self.sigmas[i])
                    all_off.extend(off)
                
                # Обновляем популяцию
                self.population = np.array([ind['x'] for ind in all_off])
                self.sigmas = np.array([ind['sigma'] for ind in all_off])
                self.function_values = np.array([ind['fitness'] for ind in all_off])
                
                # Проверка границ
                self._validate_population()
            else:
                # Обновляем суррогатную модель не каждый шаг
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
                    new_val.append(self.func(best[0]))
                
                # Отбор лучших
                comb_pop = np.vstack((self.population, np.array(new_pop)))
                comb_sig = np.vstack((self.sigmas, np.array(new_sig)))
                comb_val = np.hstack((self.function_values, np.array(new_val)))
                
                best_idx = np.argsort(comb_val)[:self.population_size]
                self.population = comb_pop[best_idx]
                self.sigmas = comb_sig[best_idx]
                self.function_values = comb_val[best_idx]
                
                # Проверка границ
                self._validate_population()
        
        i_best = np.argmin(self.function_values)
        return self.population[i_best], self.function_values[i_best]

    def _validate_population(self):
        """Проверка соблюдения границ для всей популяции"""
        for i in range(self.population_size):
            if not (np.all(self.population[i] >= self.lower_bounds) and 
                   np.all(self.population[i] <= self.upper_bounds)):
                self.population[i] = self._enforce_bounds(self.population[i])

# def objective(x):
#     return np.sum(x**2)

# # Ограничения: x[0] ∈ [0,10], x[1] ∈ [-5,5], x[2] ∈ [1,100]
# lower_bounds = [0, 10, 1]
# upper_bounds = [10, 50, 100]

# ep = EvolutionaryProgramming(
#     func=objective,
#     dimension=3,
#     lower_bounds=lower_bounds,
#     upper_bounds=upper_bounds,
#     population_size=20,
#     t_max=100
# )

# best_solution, best_fitness = ep.run()

# print("\nРезультаты:")
# print(f"Лучшее решение: {best_solution}")
# print(f"Лучшее значение функции: {best_fitness}")
# print("\nПроверка ограничений:")
# for i in range(3):
#     print(f"x[{i}] ∈ [{lower_bounds[i]}, {upper_bounds[i]}]: "
#           f"{lower_bounds[i] <= best_solution[i] <= upper_bounds[i]}")