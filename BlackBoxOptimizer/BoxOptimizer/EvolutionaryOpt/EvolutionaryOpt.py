from typing import Callable
import numpy as np
import sys
import os

# Получаем абсолютный путь к корню проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Теперь можно делать обычные импорты
from BlackBoxOptimizer.BoxOptimizer.BaseOptimizer import BaseOptimizer
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

        # опционально — сохраняем историю, как в TestStepOpt
        self.history_to_opt_model_data.append(best_x.copy())
        self.history_from_model_data.append(np.array([best_f]))

    def getResult(self) -> np.ndarray:
        """
        Возвращаем лучший найденный вектор.
        """
        # если хотите просто в виде flat array:
        return self._to_opt_model_data.vecs.flatten()
    
