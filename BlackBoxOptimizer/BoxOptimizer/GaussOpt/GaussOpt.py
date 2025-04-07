import numpy as np

from typing import Callable

from scipy.stats import norm
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels



from ..BaseOptimizer import BaseOptimizer, OptimizedVectorData

class GaussOpt(BaseOptimizer):
    def __init__(self, seed : int, kernel : kernels, *args, **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса оптимизатора
        """
        super().__init__(*args, **kwargs)

        self.seed : int = seed
        """База генератора рандомных чисел"""
        self.model = GaussianProcessRegressor(kernel=kernel)
        """Модель кригинга"""
        self.history_to_opt_model_data : np.array = np.array(self._to_opt_model_data)
        """История данных для построения модели"""
        self.target_to_opt : bool = False
        """Цель оптимизации False-минимум True-максимум"""
        self.res_of_most_opt_vec = self._to_opt_model_data[self._main_value_index]
        """Возрат наилучшего вектора, кандидат на min/max значение функции"""
        self.bound_of_vec = GaussOpt._bound_func_()

    def _expected_improvement_max(self):
        x = np.array(self.history_to_opt_model_data).reshape(1, -1)
        mu, sigma = self.model.predict(x, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        # Избегаем деления на ноль
        with np.errstate(divide='warn'):
            improvement = mu - self.res_of_most_opt_vec
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        # Возвращаем отрицательное значение, чтобы можно было минимизировать
        return -ei[0, 0]
    


    def _expected_improvement_min(self):
        x = np.array(self.history_to_opt_model_data).reshape(1, -1)
        mu, sigma = self.model.predict(x, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        # Избегаем деления на ноль
        with np.errstate(divide='warn'):
            improvement = self.res_of_most_opt_vec - mu
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        # Возвращаем отрицательное значение, чтобы можно было минимизировать
        return -ei[0, 0]
    


    def _propose_location(self):
        if self.target_to_opt:
            res = differential_evolution(func=GaussOpt._expected_improvement_max, 
                        bounds=self.bound_of_vec,
                        args=(self.model, self.res_of_most_opt_vec))
        else:
            res = differential_evolution(func=GaussOpt._expected_improvement_min,
                        bounds=self.bound_of_vec,
                        args=(self.model, self.res_of_most_opt_vec))
        return res.x
    

    def _bound_func_(self):
        bound = np.array()
        for min_vec, max_vec in zip(self._vec[:, OptimizedVectorData.min_index], self._vec[:, OptimizedVectorData.max_index]):
            to_attach = [min_vec, max_vec]
            bound.append(to_attach)
        return bound


    def _main_calc_func(self, func: Callable[[np.ndarray], np.ndarray]):
        next_x = GaussOpt._propose_location()
        self.history_to_opt_model_data.append(next_x.copy())
        self.res_of_most_opt_vec=min(func(next_x),self.res_of_most_opt_vec)
         
                    
    def getResult(self):
        return list(self._to_opt_model_data.iterVectors())