from typing import Callable, Optional, Tuple
import numpy as np
from ..BaseOptimizer import BaseOptimizer
from .OptClass import GeneticAlgorithmOptimizer

class Genetic_Algo(BaseOptimizer):
    def __init__(self, seed: int, dimension: int = 5, population_size: int = 100, 
                 generations: int = 150, init_mutation: float = 0.5, 
                 min_mutation: float = 0.2, elite_size: int = 5,
                 output_lower_bounds: Optional[np.ndarray] = None,
                 output_upper_bounds: Optional[np.ndarray] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.dimension = dimension
        self.population_size = population_size
        self.generations = generations
        self.init_mutation = init_mutation
        self.min_mutation = min_mutation
        self.elite_size = elite_size
        self.output_lower_bounds = output_lower_bounds
        self.output_upper_bounds = output_upper_bounds
        self.best_output_values = None

    def configure(self, **kwargs):
        super().configure(**kwargs)
        if 'dimension' in kwargs:
            self.dimension = kwargs['dimension']
        if 'output_lower_bounds' in kwargs:
            self.output_lower_bounds = kwargs['output_lower_bounds']
        if 'output_upper_bounds' in kwargs:
            self.output_upper_bounds = kwargs['output_upper_bounds']

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
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
        
        # Создаем обертку для функции, чтобы обрабатывать несколько выходов
        def wrapped_func(x: np.ndarray) -> np.ndarray:
            result = func(x)
            if not isinstance(result, np.ndarray):
                result = np.array([result])
            return result
        
        ga = GeneticAlgorithmOptimizer(
            func=wrapped_func,
            dimension=self._to_model_vec_size,
            population_size=self.population_size,
            generations=self.generations,
            init_mutation=self.init_mutation,
            min_mutation=self.min_mutation,
            elite_size=self.elite_size,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            output_lower_bounds=self.output_lower_bounds,
            output_upper_bounds=self.output_upper_bounds
        )
        
        # Запускаем оптимизацию (теперь возвращает три значения)
        best_x, best_f, best_outputs = ga.run()
        self.best_output_values = best_outputs

        # Записываем best_x в контейнер BaseOptimizer
        for to_vec in self._to_opt_model_data.iterVectors():
            to_vec[:] = best_x

    def getResult(self) -> np.ndarray:
        historical_data = self.getHistoricalData("vec_to_model")
        if historical_data and len(historical_data) > 0:
            return historical_data[-1]
        return np.array([])