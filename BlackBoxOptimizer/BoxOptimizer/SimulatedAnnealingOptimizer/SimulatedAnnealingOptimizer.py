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
    Улучшенная реализация алгоритма имитации отжига
    """

    def __init__(self, 
                 to_model_vec_size: int,
                 from_model_vec_size: int,
                 iter_limit: int,
                 seed: int = None,
                 initial_temp: float = 100.0,
                 min_temp: float = 1e-8,
                 cooling_rate: float = 0.99,
                 step_size: float = 1.0,
                 n_restarts: int = 3,
                 **kwargs) -> None:
        
        super().__init__(
            to_model_vec_size=to_model_vec_size,
            from_model_vec_size=from_model_vec_size,
            iter_limit=iter_limit,
            seed=seed
        )
        
        # Параметры алгоритма
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.initial_step_size = step_size
        self.step_size = step_size
        self.n_restarts = n_restarts
        
        # Текущее состояние
        self.best_solution = None
        self.best_energy = float('inf')
        self.current_energy = float('inf')
        self.restart_counter = 0
        self.iteration_counter = 0
        
        # Адаптивные параметры
        self.acceptance_rate = 0.5
        self.target_acceptance = 0.4
        
        # Инициализация нескольких кандидатов
        self._vec_candidates_size = 5  # Число параллельных цепочек

    def _init_model_vecs(self) -> None:
        """Переопределяем инициализацию для нескольких кандидатов"""
        super()._init_model_vecs()
        self._to_opt_model_data.setVectorRandValByLimits()

    def modelOptimize(self, func: Callable[[np.array], np.array]) -> None:
        """Основной метод оптимизации с поддержкой рестартов"""
        for restart in range(self.n_restarts):
            self._single_run_optimization(func)
            
            # Рестарт с лучшего решения
            if restart < self.n_restarts - 1:
                self._prepare_restart()
                
    def _single_run_optimization(self, func: Callable[[np.array], np.array]) -> None:
        """Одиночный прогон алгоритма"""
        # Инициализация нескольких цепочек
        current_solutions = [vec.copy() for vec in self._to_opt_model_data.iterVectors()]
        current_energies = [func(sol)[self._main_value_index] for sol in current_solutions]
        
        # Находим лучшую цепочку
        best_idx = np.argmin(current_energies)
        self.best_solution = current_solutions[best_idx].copy()
        self.best_energy = current_energies[best_idx]
        
        while self.iteration_counter < self._iteration_limitation and self.current_temp > self.min_temp:
            accepted = 0
            total = 0
            
            # Обновляем все цепочки
            for i in range(len(current_solutions)):
                new_solution = self._generate_neighbor(current_solutions[i])
                new_energy = func(new_solution)[self._main_value_index]
                total += 1
                
                if self._accept_solution(current_energies[i], new_energy):
                    current_solutions[i] = new_solution.copy()
                    current_energies[i] = new_energy
                    accepted += 1
                    
                    if new_energy < self.best_energy:
                        self.best_solution = new_solution.copy()
                        self.best_energy = new_energy
            
            # Адаптивное управление температурой и шагом
            self._adapt_parameters(accepted / max(total, 1))
            self.iteration_counter += 1
            
            # Периодически синхронизируем цепочки
            if self.iteration_counter % 50 == 0:
                self._synchronize_chains(current_solutions, current_energies)

    def _generate_neighbor(self, solution: np.array) -> np.array:
        """Генерация соседнего решения с адаптивным шагом"""
        neighbor = solution.copy()
        perturbation = np.random.normal(
            scale=self.step_size * (1 + 0.1 * np.random.randn()),
            size=len(solution)
        )
        
        # Применяем возмущение
        neighbor += perturbation
        
        # Применение ограничений с отражением от границ
        for i in range(len(neighbor)):
            item_props = self._to_opt_model_data._values_properties_list[i]
            
            if neighbor[i] < item_props.min:
                neighbor[i] = 2 * item_props.min - neighbor[i]
            elif neighbor[i] > item_props.max:
                neighbor[i] = 2 * item_props.max - neighbor[i]
                
            # Гарантируем, что остались в пределах
            neighbor[i] = np.clip(neighbor[i], item_props.min, item_props.max)
        
        return neighbor

    def _accept_solution(self, current_energy: float, new_energy: float) -> bool:
        """Критерий принятия решения с защитой от переполнения"""
        if new_energy < current_energy:
            return True
        
        delta = new_energy - current_energy
        if delta == 0:
            return np.random.random() < 0.5
        
        try:
            probability = math.exp(-delta / max(self.current_temp, 1e-100))
            return np.random.random() < probability
        except OverflowError:
            return False

    def _adapt_parameters(self, current_acceptance: float):
        """Адаптация температуры и размера шага"""
        # Плавное уменьшение температуры
        self.current_temp = self.initial_temp * math.exp(
            -10 * self.iteration_counter / self._iteration_limitation)
        
        # Адаптация размера шага
        if current_acceptance > self.target_acceptance:
            self.step_size *= 1.1
        else:
            self.step_size *= 0.9
            
        # Гарантируем разумные пределы для шага
        self.step_size = np.clip(self.step_size, 1e-4, 10 * self.initial_step_size)

    def _synchronize_chains(self, solutions, energies):
        """Обмен информацией между цепочками"""
        best_idx = np.argmin(energies)
        for i in range(len(solutions)):
            if i != best_idx and np.random.random() < 0.1:
                # Частичное заимствование от лучшей цепочки
                mix_mask = np.random.rand(len(solutions[i])) < 0.3
                solutions[i][mix_mask] = solutions[best_idx][mix_mask]
                energies[i] = float('inf')  # Пересчитается на следующей итерации

    def _prepare_restart(self):
        """Подготовка к рестарту алгоритма"""
        self.current_temp = self.initial_temp * 0.5  # Начинаем с меньшей температуры
        self.step_size = self.initial_step_size * 0.5
        self.iteration_counter = 0
        
        # Инициализируем новые цепочки вокруг лучшего решения
        for i, vec in enumerate(self._to_opt_model_data.iterVectors()):
            if i == 0:
                vec[:] = self.best_solution
            else:
                vec[:] = self._generate_neighbor(self.best_solution)

    def getResult(self) -> np.array:
        return self.best_solution