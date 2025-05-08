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

    def _get_bounds(self, vec_dir: str):
        """Получаем ограничения для указанного направления"""
        vec_data = self._to_opt_model_data if vec_dir == "to_model" else self._from_model_data
        size = self._to_model_vec_size if vec_dir == "to_model" else self._from_model_vec_size
        
        lower = np.full(size, -np.inf)
        upper = np.full(size, np.inf)
        
        # Безопасный доступ к границам
        for i in range(min(size, vec_data._vec.shape[0])):
            lower[i] = vec_data._vec[i, 0]  # min_index
            upper[i] = vec_data._vec[i, 1]  # max_index
            
        return lower, upper

    def modelOptimize(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        # Получаем ограничения для входных и выходных переменных
        input_lower, input_upper = self._get_bounds("to_model")
        output_lower, output_upper = self._get_bounds("from_model")
        
        # Для выходных переменных исключаем целевую функцию (первый элемент)
        output_lower = output_lower[1:] if len(output_lower) > 1 else []
        output_upper = output_upper[1:] if len(output_upper) > 1 else []
        
        ep = EvolutionaryProgramming(
            func=func,
            dimension=self._to_model_vec_size,
            population_size=self.population_size,
            offspring_per_parent=self.offspring_per_parent,
            mutation_prob=self.mutation_prob,
            sigma_init=self.sigma_init,
            t_max=self._iteration_limitation,
            lower_bounds=input_lower,
            upper_bounds=input_upper,
            output_lower_bounds=output_lower,
            output_upper_bounds=output_upper
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