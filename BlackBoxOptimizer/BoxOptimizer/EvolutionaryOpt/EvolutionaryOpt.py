from typing import Callable
import numpy as np
import sys
import os

from ..BaseOptimizer import BaseOptimizer
from .internal_ep import EvolutionaryProgramming


class EvolutionaryOpt(BaseOptimizer):

    def __init__(self, seed: int, dimension: int = 10, population_size: int = 10,
                 offspring_per_parent: int = 5, mutation_prob: float = 0.1,
                 sigma_init: float = 0.1, t_max: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.dimension = dimension
        self.population_size = population_size
        self.offspring_per_parent = offspring_per_parent
        self.mutation_prob = mutation_prob
        self.sigma_init = sigma_init
        self.t_max = t_max

    def configure(self, **kwargs):
        super().configure(**kwargs)  # Обработка параметров в базовом классе
        if 'dimension' in kwargs:
            self.dimension = kwargs['dimension']

    def modelOptimize(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Переопределяем, потому что EP запускает внешнюю модель внутри своего цикла.
        """
        # BaseOptimizer.modelOptimize не нужен — запускаем EP сразу
        self._main_calc_func(func)

    def _main_calc_func(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Запускаем всю оптимизацию за один вызов и
        кладём лучший вектор в _to_opt_model_data для совместимости.
        """
        ep = EvolutionaryProgramming(
            func                 = func,
            dimension            = self._to_model_vec_size,
            population_size      = self.population_size,
            offspring_per_parent = self.offspring_per_parent,
            mutation_prob        = self.mutation_prob,
            sigma_init           = self.sigma_init,
            t_max                = self._iteration_limitation
        )
        best_x, best_f = ep.run()

        # Записываем best_x в контейнер BaseOptimizer
        for to_vec in self._to_opt_model_data.iterVectors():
            to_vec[:] = best_x

    def _collectIterHistoryData(self, best_x: np.ndarray, best_f: float) -> None:
        """
        Сохранение информации о текущей итерации
        Использует реализацию из базового класса BaseOptimiser
        
        Parameters:
            best_x (np.ndarray): Лучшее решение текущей итерации
            best_f (float): Лучшее значение функции цели
        """
        self._to_opt_model_data.vecs = best_x.copy()
        self._from_model_data.vecs = np.array([best_f])
        super()._collectIterHistoryData()

    def getResult(self) -> np.ndarray:
        """
        Получение результатов оптимизации
        Использует исторические данные из базового класса
        """
        historical_data = self.getHistoricalData("vec_to_model")
        if historical_data and len(historical_data) > 0:
            return historical_data[-1]  # Возвращаем последний best_x
        return np.array([])