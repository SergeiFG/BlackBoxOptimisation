from typing import Callable, List
import numpy as np
from ..BaseOptimizer import BaseOptimizer, _boolItem
from .internal_ep import EvolutionaryProgramming

class EvolutionaryOpt(BaseOptimizer):
    def __init__(self, seed: int, dimension: int = 10, population_size: int = 10,
                 offspring_per_parent: int = 5, mutation_prob: float = 0.1,
                 sigma_init: float = 0.1, t_max: int = 100, 
                 discrete_indices: List[int] = None, *args, **kwargs):
        print("EvolutionaryOpt params:", population_size, offspring_per_parent, mutation_prob, sigma_init, t_max)
        # Выделяем параметры для BaseOptimizer
        base_kwargs = {
            'to_model_vec_size': kwargs.pop('to_model_vec_size', dimension),
            'from_model_vec_size': kwargs.pop('from_model_vec_size', 1),
            'iter_limit': kwargs.pop('iter_limit', t_max),
            'seed': seed
        }
        super().__init__(**base_kwargs)
        
        # Остальные параметры для EvolutionaryOpt
        self.dimension = dimension
        self.population_size = population_size
        self.offspring_per_parent = offspring_per_parent
        self.mutation_prob = mutation_prob
        self.sigma_init = sigma_init
        self.t_max = t_max
        self.discrete_indices = discrete_indices if discrete_indices is not None else []

    def configure(self, **kwargs):
        super().configure(**kwargs)
        if 'dimension' in kwargs:
            self.dimension = kwargs['dimension']
        if 'discrete_indices' in kwargs:
            self.discrete_indices = kwargs['discrete_indices']

    def _get_bounds(self, vec_dir: str):
        """Получаем ограничения для указанного направления"""
        vec_data = self._to_opt_model_data if vec_dir == "to_model" else self._from_model_data
        size = self._to_model_vec_size if vec_dir == "to_model" else self._from_model_vec_size
        
        lower = np.full(size, -np.inf)
        upper = np.full(size, np.inf)
        
        # Безопасный доступ к границам через properties_list
        for i in range(size):
            if i < len(vec_data._values_properties_list):
                lower[i] = vec_data._values_properties_list[i].min
                upper[i] = vec_data._values_properties_list[i].max
            
        return lower, upper

    def modelOptimize(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        # Получаем ограничения для входных и выходных переменных
        input_lower, input_upper = self._get_bounds("to_model")
        output_lower, output_upper = self._get_bounds("from_model")
        
        # Для выходных переменных исключаем целевую функцию (первый элемент)
        output_lower = output_lower[1:] if len(output_lower) > 1 else np.array([])
        output_upper = output_upper[1:] if len(output_upper) > 1 else np.array([])
        
        # Получаем индексы дискретных параметров
        discrete_indices = []
        for i, prop in enumerate(self._to_opt_model_data._values_properties_list):
            if isinstance(prop, _boolItem):
                discrete_indices.append(i)
        
        # Добавляем явно указанные дискретные индексы
        discrete_indices.extend(self.discrete_indices)
        discrete_indices = list(set(discrete_indices))  # Удаляем дубликаты
        
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
            output_upper_bounds=output_upper,
            discrete_indices=discrete_indices
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