from typing import Callable
import numpy as np
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
        super().configure(**kwargs)
        if 'dimension' in kwargs:
            self.dimension = kwargs['dimension']

    def _get_bounds(self):
        """Получаем ограничения из BaseOptimizer"""
        lower = np.full(self._to_model_vec_size, -np.inf)
        upper = np.full(self._to_model_vec_size, np.inf)
        
        for i in range(self._to_model_vec_size):
            lower[i] = self._to_opt_model_data._vec[i, 0]  # min_index
            upper[i] = self._to_opt_model_data._vec[i, 1]  # max_index
            
        return lower, upper

    def modelOptimize(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        # Получаем ограничения из BaseOptimizer
        lower_bounds, upper_bounds = self._get_bounds()
        
        ep = EvolutionaryProgramming(
            func=func,
            dimension=self._to_model_vec_size,
            population_size=self.population_size,
            offspring_per_parent=self.offspring_per_parent,
            mutation_prob=self.mutation_prob,
            sigma_init=self.sigma_init,
            t_max=self._iteration_limitation,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds
        )
        
        best_x, best_f = ep.run()

        # Записываем best_x в контейнер BaseOptimizer
        for to_vec in self._to_opt_model_data.iterVectors():
            to_vec[:] = best_x

    def getResult(self) -> np.ndarray:
        historical_data = self.getHistoricalData("vec_to_model")
        if historical_data and len(historical_data) > 0:
            return historical_data[-1]
        return np.array([])