import numpy as np

from typing import Callable, List


from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor


from ..BaseOptimizer import BaseOptimizer, _boolItem

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
        """Ограничения параметров векторов на выход в виде массива"""
        self.discrete_indices = []
        """Индексы дискретный параметров"""

    
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
                       method='Nelder-Mead',
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
            first_vec = self._to_opt_model_data.vecs.reshape(1, self._to_model_vec_size)
            length = first_vec.shape[0]
            factors = np.random.uniform(0.97,1.03, size=(self._to_model_vec_size*population,length))
            init_vecs = factors * first_vec

            for vec in init_vecs:
                vec[self.discrete_indices] = np.random.rand()

            return init_vecs
    """Создание первой популяции векторов для нормальной работы метода, необходимо 10*количество MV"""

    def _main_calc_func(self, func: Callable[[np.ndarray], np.ndarray]):
        self.model.fit(self.history_to_opt_model_data,self.res_history_to_opt_model_data)
        next_x = self._propose_location()
        next_x_for_fun = next_x.copy()
        for idx in self.discrete_indices:
            next_x_for_fun[idx] = 1 if next_x_for_fun[idx]>=0.5 else 0
        self.history_to_opt_model_data = np.vstack([self.history_to_opt_model_data, next_x.copy()])
        output_value = func(next_x_for_fun)
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
        if 'discrete_indices' in kwargs:
            self.discrete_indices = kwargs['discrete_indices']
    """Настройка метода"""

    def modelOptimize(self, func : Callable[[np.array], np.array]) -> None:

        discrete_indices = []
        for i, prop in enumerate(self._to_opt_model_data._values_properties_list):
            if isinstance(prop, _boolItem):
                discrete_indices.append(i)
        
        # Добавляем явно указанные дискретные индексы
        discrete_indices.extend(self.discrete_indices)
        self.discrete_indices = list(set(discrete_indices))  # Удаляем дубликаты

        
        self.history_to_opt_model_data = self._init_vecs(10)
        #print(self.history_to_opt_model_data)
        history_for_fun = self.history_to_opt_model_data
        for vec in history_for_fun:
            for idx in self.discrete_indices:
                vec[idx] = 1 if vec[idx]>=0.5 else 0

        res_list = [func(vec)[0] for vec in history_for_fun]
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
            result = self.history_to_opt_model_data[self.res_history_to_opt_model_data.index(max(self.res_history_to_opt_model_data))]
        else: 
            result = self.history_to_opt_model_data[self.res_history_to_opt_model_data.index(min(self.res_history_to_opt_model_data))]
        for idx in self.discrete_indices:
            result[idx] = 1 if result[idx]>=0.5 else 0
        return result
    """Функция результата, возращает точку"""
    