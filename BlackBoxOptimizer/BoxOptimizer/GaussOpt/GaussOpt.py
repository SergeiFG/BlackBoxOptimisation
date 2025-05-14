import numpy as np

from typing import Callable

import sys

from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler


from ..BaseOptimizer import BaseOptimizer, OptimizedVectorData

class GaussOpt(BaseOptimizer):
    def __init__(self, kernel_cfg: tuple[str, dict] = ('Matern', {'nu': 2.5}), *args, **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса оптимизатора
        """
        super().__init__(*args, **kwargs)

        # Динамически подгружаем нужное ядро
        kernel_name, kernel_params = kernel_cfg
        module = __import__('sklearn.gaussian_process.kernels', fromlist=[''])
        kernel_cls = getattr(module, kernel_name)
        kernel = kernel_cls(**kernel_params)

        self.model = GaussianProcessRegressor(kernel=kernel)
        """Модель кригинга, суррогатная модель"""
        self.history_to_opt_model_data : list = []
        """История данных для построения модели"""
        self.res_history_to_opt_model_data : list = []
        """История резльтатов функции для построения модели"""
        self.target_to_opt : bool = False
        """Цель оптимизации False-минимум True-максимум"""
        self.res_of_most_opt_vec : float = 0.0
        """Возрат наилучшего вектора, кандидат на min/max значение функции"""
        self.most_opt_vec = []
        """Кандидат на самый оптимальный вектор"""
        self.input_bound_of_vec = self._bound_func_("to_model")
        """Ограничения параметров векторов на вход в виде массива"""
        self.output_bound_of_vec = self._bound_func_("from_model")

    
    @staticmethod
    def _expected_improvement(x, model, y_opt, maximize):
        x = np.array(x).reshape(1, -1)
        mu, sigma = model.predict(x, return_std=True)
        sigma = sigma.reshape(-1, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            if maximize:
                improvement = mu - y_opt
            else:
                improvement = y_opt - mu
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma.flatten() == 0.0] = 0.0
        return -float(ei[0, 0])
    """Функция максимального правдоподобия"""

    def _propose_location(self):
        y_opt = self.res_of_most_opt_vec
        maximize = self.target_to_opt
        func = lambda x: self._expected_improvement(x, self.model, y_opt, maximize)
        res = minimize(fun=func,
                       bounds=self.input_bound_of_vec,
                       x0=self.most_opt_vec,
                       method='L-BFGS-B',
                       tol=1e-6
                       )
        return res.x
    """Функция высчитывающая следущую точку для подсчета"""

    def _bound_func_(self, vec_dir: str):
        bounds = np.array([])
        vec_data = self._to_opt_model_data if vec_dir == "to_model" else self._from_model_data
        size = self._to_model_vec_size if vec_dir == "to_model" else self._from_model_vec_size
        
        lower = np.full(size, -np.inf)
        upper = np.full(size, np.inf)
        
        # Безопасный доступ к границам через properties_list
        for i in range(size):
            if i < len(vec_data._values_properties_list):
                lower[i] = vec_data._values_properties_list[i].min
                upper[i] = vec_data._values_properties_list[i].max
    
        for min_v, max_v in zip(lower, upper):
            bounds = np.append(bounds, (min_v, max_v), axis=0)

        bounds = bounds.reshape(size, 2) 
        return bounds
    """Функция возращяющая массив минимумов и максимумов вида [[min1, max1],[min2, max2]...]"""

    def _penalize_fitness(self, fitness, output_values):
        """Штрафование fitness при нарушении ограничений выходных переменных"""
        penalty = 0
        output_params = output_values[1:] if len(output_values) > 1 else []
        
        for i in range(min(len(output_params), len(self.output_bound_of_vec))):
            if not np.isinf(self.output_bound_of_vec[i][0]):
                penalty += max(self.output_bound_of_vec[i][0] - output_params[i], 0)**2
                
        for i in range(min(len(output_params), len(self.output_bound_of_vec))):
            if not np.isinf(self.output_bound_of_vec[i][1]):
                penalty += max(output_params[i] - self.output_bound_of_vec[i][1], 0)**2
                
        return fitness + 1e6 * penalty

    def _init_vecs(self,population):
            if self._seed is not None:
                np.random.seed(self._seed)
            first_vec = np.array(self._to_opt_model_data._vec[:,OptimizedVectorData.values_index_start].copy())
            length = first_vec.shape[0]
            factors = np.random.uniform(0.99,1.01, size=(self._to_model_vec_size*population,length))
            init_vecs = factors * first_vec

            return [first_vec] + [init_vecs[i] for i in range(init_vecs.shape[0])]
    """Создание первой популяции векторов для нормальной работы метода, необходимо 10*количество MV"""

    def _main_calc_func(self, func: Callable[[np.ndarray], np.ndarray]):
        self.model.fit(self.history_to_opt_model_data,self.res_history_to_opt_model_data)
        next_x = self._propose_location()
        self.history_to_opt_model_data.append(next_x.copy())
        output_value = func(next_x.copy())
        candidate_vec = output_value[0]
        if not self._check_output_constraints(output_value):
                candidate_vec = self._penalize_fitness(candidate_vec, output_value)
        self.res_history_to_opt_model_data.append(candidate_vec)
        if self.target_to_opt:
            self.res_of_most_opt_vec = max(candidate_vec, self.res_of_most_opt_vec)
        else:
            self.res_of_most_opt_vec = min(candidate_vec, self.res_of_most_opt_vec)
        
        self.most_opt_vec = self.history_to_opt_model_data[self.res_history_to_opt_model_data.index(self.res_of_most_opt_vec)]
        
    """Основная функция подсчета"""

    def _check_output_constraints(self, output_values):
        """Проверка ограничений выходных переменных"""
        if len(output_values) <= 1:  # Только целевая функция
            return True
            
        # Проверяем только те параметры, для которых заданы ограничения
        num_output_params = min(len(self.output_bound_of_vec), len(output_values)-1)
        
        for i in range(num_output_params):
            if (output_values[i+1] < self.output_bound_of_vec[i][0] or 
                output_values[i+1] > self.output_bound_of_vec[i][1]):
                return False
      
        return True

    def configure(self, **kwargs):
        kernel_cfg = kwargs.pop('kernel_cfg', None)

        super().configure(**kwargs)

        if kernel_cfg is not None:
            kernel_name, kernel_params = kernel_cfg

            module = __import__('sklearn.gaussian_process.kernels', fromlist=[''])
            kernel_cls = getattr(module, kernel_name)

            kernel = kernel_cls(**kernel_params)

            self.model = GaussianProcessRegressor(kernel=kernel)
    """Настройка метода"""

    def modelOptimize(self, func : Callable[[np.array], np.array]) -> None:
        self.history_to_opt_model_data = self._init_vecs(10)
        res_list = [func(vec)[0] for vec in self.history_to_opt_model_data]
        self.res_history_to_opt_model_data = res_list
        
        if self.target_to_opt:
            self.res_of_most_opt_vec = max(res_list)
        else:
            self.res_of_most_opt_vec = min(res_list)

        self.most_opt_vec = self.history_to_opt_model_data[self.res_history_to_opt_model_data.index(self.res_of_most_opt_vec)]

        for _ in range(self._iteration_limitation):
            self.input_bound_of_vec = self._bound_func_("to_model")
            self.output_bound_of_vec = self._bound_func_("from_model")
            self._main_calc_func(func=func)
            
    """Функция инициализации и оптимизации"""

    def getResult(self):
        if self.target_to_opt:
            return self.history_to_opt_model_data[self.res_history_to_opt_model_data.index(max(self.res_history_to_opt_model_data))]
        else: 
            return self.history_to_opt_model_data[self.res_history_to_opt_model_data.index(min(self.res_history_to_opt_model_data))]
    """Функция результата, возращает точку"""
    