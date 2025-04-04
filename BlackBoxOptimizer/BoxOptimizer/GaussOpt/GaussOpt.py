import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels



from ..BaseOptimizer import BaseOptimizer

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

    def _expected_improvement_max(self):
        x = np.array(self._to_opt_model_data).reshape(1, -1)
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
        x = np.array(x).reshape(1, -1)
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
    


    def propose_location(self):
        if self.target_to_opt:
            res = differential_evolution(func=GaussOpt._expected_improvement_max, 
                        bounds=bounds,
                        args=(self.model, self.res_of_most_opt_vec))
        else:
            res = differential_evolution(func=GaussOpt._expected_improvement_min,
                        bounds=bounds,
                        args=(self.model, self.res_of_most_opt_vec))
        return res.x
    

    def _main_calc_func(self, func):
        next_x = GaussOpt.propose_location()
        self.history_to_opt_model_data.append(next_x)
        self.res_of_most_opt_vec=min(func(next_x),self.res_of_most_opt_vec)
                    
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels



from ..BaseOptimizer import BaseOptimizer

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

    def _expected_improvement_max(self):
        x = np.array(self._to_opt_model_data).reshape(1, -1)
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
        x = np.array(x).reshape(1, -1)
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
    


    def propose_location(self):
        if self.target_to_opt:
            res = differential_evolution(func=GaussOpt._expected_improvement_max, 
                        bounds=bounds,
                        args=(self.model, self.res_of_most_opt_vec))
        else:
            res = differential_evolution(func=GaussOpt._expected_improvement_min,
                        bounds=bounds,
                        args=(self.model, self.res_of_most_opt_vec))
        return res.x
    

    def _main_calc_func(self, func):
        next_x = GaussOpt.propose_location()
        self.history_to_opt_model_data.append(next_x)
        self.res_of_most_opt_vec=min(func(next_x),self.res_of_most_opt_vec)
                    
