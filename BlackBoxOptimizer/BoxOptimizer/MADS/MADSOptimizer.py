import PyNomad
import numpy as np
from typing import Callable

from ..BaseOptimizer import BaseOptimizer


class MADSOptimizer(BaseOptimizer):
    def __init__(self, to_model_vec_size, from_model_vec_size, iter_limit, main_value_index = 0):
       super().__init__(to_model_vec_size, from_model_vec_size, iter_limit, main_value_index)
       self._nomad_params = ["DIMENSION " + str(to_model_vec_size),
                             "BB_OUTPUT_TYPE OBJ",
                             "MAX_BB_EVAL " + str(iter_limit),
                             "DISPLAY_DEGREE 2",
                             "DISPLAY_ALL_EVAL true",
                             "DISPLAY_STATS BBE OBJ",
                             "HISTORY_FILE history0.txt"]
       self._lb = None
       self._ub = None
       self._best_solution = None
       self._best_value = np.inf

    def _main_calc_func(self):
        """Итерации выполняются внутри MADS"""
        pass
    def _calc_objective_function_value(self):
        pass

    def modelOptimize(self, func: Callable[[np.array], np.array]) -> None:
       
        params = self._nomad_params.copy()
        lb = list(self._to_opt_model_data._vec[:, OptimizedVectorData.min_index])
        ub = list(self._to_opt_model_data._vec[:, OptimizedVectorData.max_index])
        x0 = list(next(self._to_opt_model_data.iterVectors()))
        
        def nomad_blackbox(x):
            x_array = np.array(x, dtype = float)
            res = func(x_array)
            return [float(val) for val in res]
        
        result = PyNomad.optimize(
            nomad_blackbox,
            x0,
            lb,
            ub,
            params
        )
        if result and 'x_best' in result:
            self._best_solution = np.array(result['x_best'])
            self._best_value = result['f_best']
    
    def getResult(self):
        if self._best_solution is None:
            raise RuntimeError("Jgnbvbpfwbz yt ,skf dsgjkytyf")
        return self._best_solution, self._best_value
