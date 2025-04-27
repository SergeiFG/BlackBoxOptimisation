from typing import Callable
import numpy as np
from ..BaseOptimizer import BaseOptimizer
from .OptClass import GeneticAlgorithmOptimizer

class Genetic_Algo(BaseOptimizer):
    def __init__(self, seed: int, dimension:int = 5, population_size:int = 100, generations:int = 150,
                 init_mutation:float = 0.5, min_mutation:float = 0.2, elite_size:int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.dimension = dimension
        self.population_size = population_size
        self.generations = generations
        self.init_mutation = init_mutation
        self.min_mutation = min_mutation
        self.elite_size = elite_size

       
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
        
        ga = GeneticAlgorithmOptimizer(
            func=func,
            dimension=self._to_model_vec_size,
            population_size=self.population_size,
            generations = self.generations,
            init_mutation = self.init_mutation,
            min_mutation = self.min_mutation,
            elite_size = self.elite_size,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds
        )
        
        best_x, best_f = ga.run()

        # Записываем best_x в контейнер BaseOptimizer
        for to_vec in self._to_opt_model_data.iterVectors():
            to_vec[:] = best_x

    def getResult(self) -> np.ndarray:
        historical_data = self.getHistoricalData("vec_to_model")
        if historical_data and len(historical_data) > 0:
            return historical_data[-1]
        return np.array([])