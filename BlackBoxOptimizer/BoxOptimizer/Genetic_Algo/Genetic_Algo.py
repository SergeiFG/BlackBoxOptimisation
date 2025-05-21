from typing import Callable, Optional, List
import numpy as np
from ..BaseOptimizer import BaseOptimizer
from .OptClass import GeneticAlgorithmOptimizer

class Genetic_Algo(BaseOptimizer):
    def __init__(self, seed: int, dimension: int = 5, population_size: int = 100, 
                 generations: int = 150, init_mutation: float = 0.5, 
                 min_mutation: float = 0.2, elite_size: int = 5,
                 output_lower_bounds: Optional[np.ndarray] = None,
                 output_upper_bounds: Optional[np.ndarray] = None,
                 discrete_indices: Optional[List[int]] = None,
                 *args, **kwargs):
        base_kwargs = {
            'to_model_vec_size': kwargs.pop('to_model_vec_size', dimension),
            'from_model_vec_size': kwargs.pop('from_model_vec_size', 1),
            'iter_limit': kwargs.pop('iter_limit', generations),
            'seed': seed
        }
        super().__init__(**base_kwargs)
        self.seed = seed
        self.dimension = dimension
        self.population_size = population_size
        self.generations = generations
        self.init_mutation = init_mutation
        self.min_mutation = min_mutation
        self.elite_size = elite_size
        self.output_lower_bounds = output_lower_bounds
        self.output_upper_bounds = output_upper_bounds
        self.discrete_indices = discrete_indices if discrete_indices is not None else []
        self.best_output_values = None

    def configure(self, **kwargs):
        super().configure(**kwargs)
        for key in ['dimension', 'output_lower_bounds', 'output_upper_bounds', 'discrete_indices']:
            if key in kwargs:
                setattr(self, key, kwargs[key])

    def _get_bounds(self):
        """Получение границ из BaseOptimizer"""
        lower = np.array([prop.min for prop in self._to_opt_model_data._values_properties_list])
        upper = np.array([prop.max for prop in self._to_opt_model_data._values_properties_list])
        return lower, upper

    def modelOptimize(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        lower_bounds, upper_bounds = self._get_bounds()
        ga = GeneticAlgorithmOptimizer(
            func=func,
            dimension=self._to_model_vec_size,
            population_size=self.population_size,
            generations=self.generations,
            init_mutation=self.init_mutation,
            min_mutation=self.min_mutation,
            elite_size=self.elite_size,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            output_lower_bounds=self.output_lower_bounds,
            output_upper_bounds=self.output_upper_bounds,
            discrete_indices=self.discrete_indices
        )
        best_x, best_f, best_outputs = ga.run()
        for to_vec in self._to_opt_model_data.iterVectors():
            to_vec[:] = best_x
        self.best_output_values = best_outputs

    def getResult(self) -> np.ndarray:
        historical_data = self.getHistoricalData("vec_to_model")
        return historical_data[-1] if historical_data else np.array([])