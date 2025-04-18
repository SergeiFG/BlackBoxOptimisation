import math
import numpy as np
from typing import Callable,Tuple
from ..BaseOptimizer import BaseOptimizer,OptimizedVectorData

class SimulatedAnnealingOptimizer(BaseOptimizer):
    def __init__(self,
                 to_model_vec_size: int,
                 from_model_vec_size: int,
                 iter_limit: int,
                 initial_temperature: float = 100.0,
                 cooling_rate: float = 0.9,
                 main_value_index: int = 0,
                 maximize: bool = False  # Новый параметр (по умолчанию ищем минимум)
                 ) -> None:

        super().__init__(
            to_model_vec_size=to_model_vec_size,
            from_model_vec_size=from_model_vec_size,
            iter_limit=iter_limit,
            main_value_index=main_value_index
        )
        self._temperature = initial_temperature
        self._cooling_rate = cooling_rate
        self._current_energy = None 
        self._best_vector = None     
        self._best_energy = None  
        self._maximize = maximize  # Сохраняем режим оптимизации  

    def _init_param(self):
        """Переопределение метода для установки 1 кандидата (текущий вектор)."""
        self._vec_candidates_size = 1
        super()._init_param()

    def _generate_neighbor(self) -> np.array:
        current_vector = next(self._to_opt_model_data.iterVectors()).copy()
        neighbor = current_vector + np.random.normal(
            scale=0.1 * self._temperature,  # Масштаб зависит от температуры
            size=len(current_vector)
        )
        for i in range(len(neighbor)):
            min_val = self._to_opt_model_data._vec[i][OptimizedVectorData.min_index]
            max_val = self._to_opt_model_data._vec[i][OptimizedVectorData.max_index]
            neighbor[i] = np.clip(neighbor[i], min_val, max_val)
        return neighbor

    def _accept_solution(self, new_energy: float) -> bool:
        """Определить, принять ли новое решение (с учетом температуры)."""
        if self._maximize:
            # Для максимизации: принимаем решения с большей энергией
            if new_energy > self._current_energy:
                return True
            probability = math.exp((new_energy - self._current_energy) / self._temperature)
        else:
            # Для минимизации (старый вариант)
            if new_energy < self._current_energy:
                return True
            probability = math.exp(-(new_energy - self._current_energy) / self._temperature)
        return np.random.random() < probability

    def _main_calc_func(self) -> None:
        # Получаем текущий вектор и его энергию (значение целевой функции)
        current_vector = next(self._to_opt_model_data.iterVectors())
        self._current_energy = self._from_model_data._vec[self._main_value_index][OptimizedVectorData.values_index_start]

        # Генерируем соседнее решение
        neighbor = self._generate_neighbor()

        # Сохранение нового кандидата
        self._to_opt_model_data._vec[:, OptimizedVectorData.values_index_start] = neighbor

        # Для первой итерации
        if self._best_vector is None:
            self._best_vector = current_vector.copy()
            self._best_energy = self._current_energy

        # Охлаждение
        self._temperature *= self._cooling_rate

    def _calc_objective_function_value(self) -> None:
        pass  # Не требуется, так как значение вычисляется в modelOptimize

    def modelOptimize(self, func: Callable[[np.array], np.array]) -> None:
        for _ in range(self._iteration_limitation):
            # Вычисляем значение функции для текущего вектора
            current_vector = next(self._to_opt_model_data.iterVectors())
            result = func(current_vector)
            self._from_model_data._vec[:, OptimizedVectorData.values_index_start] = result

            # Обновление лучшего решения
            current_energy = result[self._main_value_index]
            if self._best_energy is None:
                self._best_energy = current_energy
                self._best_vector = current_vector.copy()
            else:
                if self._maximize:
                    if current_energy > self._best_energy:  # Для максимизации
                        self._best_energy = current_energy
                        self._best_vector = current_vector.copy()
                else:
                    if current_energy < self._best_energy:  # Для минимизации
                        self._best_energy = current_energy
                        self._best_vector = current_vector.copy()

            self._main_calc_func()

    def getResult(self) -> Tuple[np.array, float]:
        return self._best_vector, self._best_energy