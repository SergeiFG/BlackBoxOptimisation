import math
import numpy as np
from typing import Callable, Tuple, Optional
from ..BaseOptimizer import BaseOptimizer, OptimizedVectorData

from ..BaseOptimizer import BaseOptimizer
import numpy as np
from typing import Callable
import math

class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    SimulatedAnnealingOptimizer
    ---
    Реализация алгоритма имитации отжига для оптимизации
    """
    
    def __init__(self, 
                 seed: int,
                 initial_temp: float = 100.0,
                 min_temp: float = 1e-3,
                 cooling_rate: float = 0.95,
                 step_size: float = 1.0,
                 *args, **kwargs) -> None:
        """
        Конструктор класса оптимизатора
        
        Аргументы:
            seed: База генератора случайных чисел
            initial_temp: Начальная температура
            min_temp: Минимальная температура (критерий остановки)
            cooling_rate: Скорость охлаждения
            step_size: Начальный размер шага
        """
        super().__init__(*args, **kwargs)
        
        self.seed = seed
        np.random.seed(seed)
        
        # Параметры алгоритма отжига
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        
        # История для отладки
        self.history_to_opt_model_data = []
        self.history_from_model_data = []
        self.temperature_history = []
        
        # Лучшее найденное решение
        self.best_solution = None
        self.best_energy = float('inf')

    def modelOptimize(self, func: Callable[[np.array], np.array]) -> None:
        """
        Запуск оптимизации через передачу функции черного ящика
        """
        # Инициализация начального решения
        current_solution = next(self._to_opt_model_data.iterVectors()).copy()
        current_energy = func(current_solution)[0]
        
        self.best_solution = current_solution.copy()
        self.best_energy = current_energy
        
        for _ in range(self._iteration_limitation):
            if self.current_temp <= self.min_temp:
                break
                
            # Генерация нового решения
            new_solution = self._generate_neighbor(current_solution)
            
            # Вычисление энергии нового решения
            new_energy = func(new_solution)[0]
            
            # Принятие решения о переходе
            if self._accept_solution(current_energy, new_energy):
                current_solution = new_solution.copy()
                current_energy = new_energy
                
                # Обновление лучшего решения
                if new_energy < self.best_energy:
                    self.best_solution = new_solution.copy()
                    self.best_energy = new_energy
            
            # Сохранение истории
            self._save_history(current_solution, current_energy)
            
            # Охлаждение
            self.current_temp *= self.cooling_rate
            self.temperature_history.append(self.current_temp)
            
            # Адаптация размера шага
            self.step_size *= 0.99  # Постепенно уменьшаем шаг

    def _generate_neighbor(self, solution: np.array) -> np.array:
        """Генерация соседнего решения"""
        neighbor = solution.copy()
        
        # Случайное изменение с нормальным распределением
        perturbation = np.random.normal(scale=self.step_size, size=len(solution))
        neighbor += perturbation
        
        # Проверка ограничений
        for i in range(len(neighbor)):
            min_val = self._to_opt_model_data._vec[i, OptimizedVectorData.min_index]
            max_val = self._to_opt_model_data._vec[i, OptimizedVectorData.max_index]
            
            if min_val != -np.inf and neighbor[i] < min_val:
                neighbor[i] = min_val
            if max_val != np.inf and neighbor[i] > max_val:
                neighbor[i] = max_val
                
        return neighbor

    def _accept_solution(self, current_energy: float, new_energy: float) -> bool:
        """Определение, принять ли новое решение"""
        if new_energy < current_energy:
            return True
        
        # Вероятность принятия худшего решения
        delta = new_energy - current_energy
        probability = math.exp(-delta / self.current_temp)
        
        return np.random.random() < probability

    def _save_history(self, solution: np.array, energy: float) -> None:
        """Сохранение истории итераций"""
        self.history_to_opt_model_data.append(solution.copy())
        self.history_from_model_data.append(np.array([energy]))

    def getResult(self) -> np.array:
        """Возвращает лучшее найденное решение"""
        return self.best_solution