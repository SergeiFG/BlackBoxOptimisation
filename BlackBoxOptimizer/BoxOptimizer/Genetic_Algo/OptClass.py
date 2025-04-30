import numpy as np
import time

class GeneticAlgorithmOptimizer:
    def __init__(self, func, dimension=5, population_size=100, generations=150,
                 init_mutation=0.5, min_mutation=0.2, elite_size=5,
                 lower_bounds=None, upper_bounds=None):
        """
        Инициализация генетического алгоритма для оптимизации.
        
        Параметры:
        func - целевая функция для минимизации
        dimension - размерность задачи
        population_size - размер популяции
        generations - количество поколений
        init_mutation - начальная вероятность мутации
        min_mutation - минимальная вероятность мутации
        elite_size - количество элитных особей
        lower_bounds - нижние границы параметров
        upper_bounds - верхние границы параметров
        """
        self.func = func
        self.dimension = dimension
        self.population_size = population_size
        self.generations = generations
        self.init_mutation = init_mutation
        self.min_mutation = min_mutation
        self.elite_size = elite_size
        
        # Установка границ параметров
        if lower_bounds is None:
            self.lower_bounds = np.full(dimension, -np.inf)
        else:
            self.lower_bounds = np.array(lower_bounds)
            
        if upper_bounds is None:
            self.upper_bounds = np.full(dimension, np.inf)
        else:
            self.upper_bounds = np.array(upper_bounds)
            
        # Проверка корректности границ
        if len(self.lower_bounds) != dimension or len(self.upper_bounds) != dimension:
            raise ValueError("Количество границ должно совпадать с размерностью")
            
        # История для визуализации
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.mutation_rates = []
        
    def _initialize_population(self):
        """Инициализация популяции с учетом границ параметров"""
        population = np.empty((self.population_size, self.dimension))
        for i in range(self.dimension):
            low, high = self.lower_bounds[i], self.upper_bounds[i]
            population[:, i] = np.random.uniform(low, high, size=self.population_size)
        return population
    
    def _get_current_mutation_rate(self, generation):
        """Вычисление текущей вероятности мутации с экспоненциальным затуханием"""
        decay_rate = np.log(self.init_mutation/self.min_mutation) / self.generations
        return self.init_mutation * np.exp(-decay_rate * generation)
    
    # def _evaluate(self, population):
    #     """Оценка популяции"""
    #     return np.array([self.func(ind) for ind in population])
    
    def _evaluate(self, population, ask_time=100):

        """Оценка популяции.
        
        Если self.func возвращает None (функция "на паузе"), 
        метод продолжает опрашивать её каждые ask_time миллисекунд,
        пока не получит результат.
        """

        fitness_values = []
        
        for ind in population:
            while True:
                fitness = self.func(ind)
                if fitness is not None:
                    fitness_values.append(fitness)
                    break
                time.sleep(ask_time / 1000)  # Переводим мс в секунды
        
        return np.array(fitness_values)
    
    def _selection(self, population, fitness, num_parents):
        """Рулеточный отбор с учетом минимизации"""
        inverted_fitness = 1 / (1 + fitness)  # Добавляем 1 чтобы избежать деления на 0
        probabilities = inverted_fitness / np.sum(inverted_fitness)
        parent_indices = np.random.choice(
            range(len(population)), 
            size=num_parents, 
            replace=True, 
            p=probabilities
        )
        return population[parent_indices]
    
    def _crossover(self, parents, offspring_size):
        """Различные операторы кроссовера с учетом границ параметров"""
        offspring = np.empty(offspring_size)
        
        for k in range(offspring_size[0]):
            p1, p2 = np.random.choice(range(parents.shape[0]), 2, replace=False)
            
            # Случайный выбор оператора кроссовера
            crossover_type = np.random.choice(
                ['arithmetic', 'blend', 'logarithmic', 'single_point'],
                p=[0.4, 0.3, 0.2, 0.1]
            )
            
            if crossover_type == 'arithmetic':
                alpha = np.random.random()
                offspring[k] = alpha*parents[p1] + (1-alpha)*parents[p2]
                
            elif crossover_type == 'blend':
                alpha = 0.5
                for i in range(self.dimension):
                    x1, x2 = sorted([parents[p1][i], parents[p2][i]])
                    diff = x2 - x1
                    low = max(x1 - alpha*diff, self.lower_bounds[i])
                    high = min(x2 + alpha*diff, self.upper_bounds[i])
                    offspring[k][i] = np.random.uniform(low, high)
                    
            elif crossover_type == 'logarithmic':
                ratio = np.abs(np.log1p(np.abs(parents[p1] / (parents[p2] + 1e-10))))
                ratio = ratio / (ratio + 1)
                offspring[k] = ratio*parents[p1] + (1-ratio)*parents[p2]
                
            else:  # single_point
                point = np.random.randint(1, self.dimension)
                offspring[k][:point] = parents[p1][:point]
                offspring[k][point:] = parents[p2][point:]
            
            # Гарантируем соблюдение границ
            for i in range(self.dimension):
                offspring[k][i] = np.clip(
                    offspring[k][i], 
                    self.lower_bounds[i], 
                    self.upper_bounds[i]
                )
        
        return offspring
    
    def _mutate(self, offspring, generation):
        """Гауссова мутация с учетом границ параметров"""
        current_mutation_rate = self._get_current_mutation_rate(generation)
        mutation_strength = 0.5 * (1 - generation/self.generations)
        
        for i in range(offspring.shape[0]):
            if np.random.random() < current_mutation_rate:
                for j in range(self.dimension):
                    mutation = np.random.normal(0, mutation_strength)
                    offspring[i][j] = np.clip(
                        offspring[i][j] + mutation,
                        self.lower_bounds[j],
                        self.upper_bounds[j]
                    )
        return offspring
    
    def _enforce_bounds(self, x):
        """Строгое соблюдение границ с защитой от численных погрешностей"""
        x_clipped = np.clip(x, self.lower_bounds, self.upper_bounds)
        for i in range(self.dimension):
            if np.isclose(x_clipped[i], self.lower_bounds[i]):
                x_clipped[i] = self.lower_bounds[i] + 1e-5
            elif np.isclose(x_clipped[i], self.upper_bounds[i]):
                x_clipped[i] = self.upper_bounds[i] - 1e-5
        return x_clipped
    
    def run(self):
        """Запуск генетического алгоритма"""
        population = self._initialize_population()
        
        for gen in range(self.generations):
            fitness = self._evaluate(population)
            current_mutation = self._get_current_mutation_rate(gen)
            self.mutation_rates.append(current_mutation)
            
            # Сохранение статистики
            best_fitness = np.min(fitness)
            avg_fitness = np.mean(fitness)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # Элитизм
            elite_indices = np.argsort(fitness)[:self.elite_size]
            elite = population[elite_indices]
            
            # Селекция и кроссовер
            parents = self._selection(population, fitness, self.population_size - self.elite_size)
            offspring = self._crossover(parents, (self.population_size - self.elite_size, self.dimension))
            
            # Мутация
            mutated_offspring = self._mutate(offspring, gen)
            population[:self.population_size-self.elite_size] = mutated_offspring
            population[self.population_size-self.elite_size:] = elite
            

        # Получение лучшего решения
        final_fitness = self._evaluate(population)
        best_idx = np.argmin(final_fitness)
        self.best_solution = population[best_idx]
        self.best_fitness = final_fitness[best_idx]
        
        return self.best_solution, self.best_fitness
    


# # Пример использования:
# if __name__ == "__main__":
#     # Определение целевой функции (как в исходном коде)
#     def target_function(x):
#         result = 0
        
#         # Шумовая компонента
#         result += np.sum( (x-1.273)**2 )
        
#         return result

#     # Границы параметров
#     param_bounds = [
#         (-5.0, 5.0),    # Параметр 1
#         (-3.0, 3.0),     # Параметр 2
#         (0.0, 10.0),     # Параметр 3
#         (-1.0, 1.0),     # Параметр 4
#         (-10.0, 0.0)     # Параметр 5
#     ]
    
#     lower_bounds = [b[0] for b in param_bounds]
#     upper_bounds = [b[1] for b in param_bounds]

#     # Создание и запуск оптимизатора
#     ga = GeneticAlgorithmOptimizer(
#         func=target_function,
#         dimension=5,
#         population_size=300,
#         generations=100,
#         init_mutation=0.5,
#         min_mutation=0.0001,
#         elite_size=10,
#         lower_bounds=lower_bounds,
#         upper_bounds=upper_bounds
#     )
    
#     best_solution, best_fitness = ga.run()
    
#     # Вывод результатов
#     print("\nЛучшее найденное решение:")
#     print("Параметр |   Значение   | Границы")
#     for i, (val, low, high) in enumerate(zip(best_solution, lower_bounds, upper_bounds)):
#         print(f"{i+1:7d} | {val:12.6f} | [{low:5.1f}, {high:5.1f}]")
#     print(f"\nЗначение функции: {best_fitness:.6f}")