import math
import numpy as np
from typing import Callable, Tuple, Optional
from ..BaseOptimizer import BaseOptimizer, OptimizedVectorData

class SimulatedAnnealingOptimizer(BaseOptimizer):
    def __init__(self,
                 to_model_vec_size: int,
                 from_model_vec_size: int,
                 iter_limit: int,
                 initial_temperature: float = 1000.0,
                 cooling_rate: float = 0.99,
                 seed: Optional[int] = None,
                 **kwargs
                 ) -> None:
        
        super().__init__(
            to_model_vec_size=to_model_vec_size,
            from_model_vec_size=from_model_vec_size,
            iter_limit=iter_limit,
            seed=seed
        )
        self._temperature = max(initial_temperature * 0.99, 0.01)
        self._cooling_rate = cooling_rate
        self._current_energy = None
        self._best_vector = None
        self._best_energy = None
        self._current_vector = None
        self._main_value_index = 0  
    def _generate_neighbor(self) -> np.array:
        """Генерация соседнего решения"""
        current_vector = next(self._to_opt_model_data.iterVectors()).copy()
        neighbor = current_vector + np.random.normal(
            scale=0.1, 
            size=len(current_vector))
        
    
        for i in range(len(neighbor)):
            min_val = self._to_opt_model_data._vec[i][OptimizedVectorData.min_index]
            max_val = self._to_opt_model_data._vec[i][OptimizedVectorData.max_index]
            neighbor[i] = np.clip(neighbor[i], min_val, max_val)
        
        return neighbor

    def _accept_solution(self, new_energy: float) -> bool:
        """Критерий принятия решения"""
        if self._current_energy is None:
            return True
            
        if new_energy < self._current_energy:
            return True
            
 
        probability = math.exp(-(new_energy - self._current_energy) / self._temperature)
        return np.random.random() < probability

    def _main_calc_func(self) -> None:
        """Основная функция расчета одной итерации"""

        if self._temperature < 1.0 and np.random.random() < 0.05:
            self._temperature *= 2 
        self._current_vector = next(self._to_opt_model_data.iterVectors())
        self._current_energy = self._from_model_data._vec[self._main_value_index][OptimizedVectorData.values_index_start]


        neighbor = self._generate_neighbor()
        self._to_opt_model_data._vec[:, OptimizedVectorData.values_index_start] = neighbor

        self._temperature *= self._cooling_rate

    def modelOptimize(self, func: Callable[[np.array], np.array]) -> Tuple[np.array, float]:
        """Основной метод оптимизации"""
        for _ in range(self._iteration_limitation):
            current_vector = next(self._to_opt_model_data.iterVectors())
            result = func(current_vector)
            self._from_model_data._vec[:, OptimizedVectorData.values_index_start] = result

            current_energy = result[self._main_value_index]
            
            if self._best_energy is None or current_energy < self._best_energy:
                self._best_energy = current_energy
                self._best_vector = current_vector.copy()

            self._main_calc_func()

        return self.getResult()

    def getResult(self) -> Tuple[np.array, float]:
        """Возвращает лучшее найденное решение и его значение"""
        if self._best_vector is None:
            current_vector = next(self._to_opt_model_data.iterVectors())
            return current_vector, self._from_model_data._vec[self._main_value_index][OptimizedVectorData.values_index_start]
        return self._best_vector, self._best_energy

    def reset(self) -> None:
        """Сброс состояния оптимизатора"""
        super().reset()
        self._temperature = self._initial_temperature
        self._current_energy = None
        self._best_vector = None
        self._best_energy = None
        self._current_vector = None