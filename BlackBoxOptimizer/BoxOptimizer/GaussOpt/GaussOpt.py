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
        mu, sigma = model.predict(x, return_std=True)   # shapes (1, m), (1, m)
        mu, sigma = mu.flatten(), sigma.flatten()       # теперь 1-D длины m

        # считаем EI только для [0]-го выхода, но храним в массиве
        ei = np.zeros_like(mu)
        imp = (mu[0] - y_opt) if maximize else (y_opt - mu[0])
        if sigma[0] > 0:
            Z = imp / sigma[0]
            ei[0] = imp * norm.cdf(Z) + sigma[0] * norm.pdf(Z)

        # безопасно зануляем EI, если σ[0]==0
        ei[0] = np.where(sigma[0] == 0.0, 0.0, ei[0])

        # возвращаем скаляр
        return -float(ei[0])
    """Функция максимального правдоподобия"""

    def _propose_location(self):
        y_opt    = self.res_of_most_opt_vec
        maximize = self.target_to_opt
        # базовая функция (EI)
        func = lambda x: self._expected_improvement(x, self.model, y_opt, maximize)

        # создаём список ограничений на каждый выход beyond bounds
        constraints = []
        # model.predict возвращает [f, g1, g2, ...], где g — вторые выходы
        for i, (lb, ub) in enumerate(self.output_bound_of_vec):
            # g_i(x) >= lb  ⇒  g_i(x) - lb ≥ 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, lb=lb:  self.model.predict(x.reshape(1, -1))[0, i] - lb
            })
            # g_i(x) ≤ ub  ⇒  ub - g_i(x) ≥ 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, ub=ub: ub - self.model.predict(x.reshape(1, -1))[0, i]
            })

        res = minimize(
            fun=func,
            x0 = self.most_opt_vec,
            bounds=self.input_bound_of_vec,
            constraints=constraints,
            method='SLSQP',
            options={'ftol':1e-6, 'maxiter':100}
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

    # def _penalize_fitness(self, fitness, output_values):
    #     """Штрафование fitness при нарушении ограничений выходных переменных"""
    #     penalty = 0
    #     output_params = output_values[1:] if len(output_values) > 1 else []
        
    #     for i in range(min(len(output_params), len(self.output_bound_of_vec))):
    #         if not np.isinf(self.output_bound_of_vec[i][0]):
    #             penalty += max(self.output_bound_of_vec[i][0] - output_params[i], 0)**2
                
    #     for i in range(min(len(output_params), len(self.output_bound_of_vec))):
    #         if not np.isinf(self.output_bound_of_vec[i][1]):
    #             penalty += max(output_params[i] - self.output_bound_of_vec[i][1], 0)**2
                
    #     return fitness + 1e2 * penalty

    def _init_vecs(self,population):
            if self._seed is not None:
                np.random.seed(self._seed)
            # Преобразуем входные данные в numpy array для удобства
            constraints = np.asarray(self.input_bound_of_vec)
            # Генерируем данные для каждой компоненты отдельно
            components = [
                np.random.uniform(low=min_max[0], high=min_max[1], size=population*self._to_model_vec_size)
                for min_max in constraints
            ]
            
            # Транспонируем результат, чтобы векторы были строками матрицы
            return np.column_stack(components)
    """Создание первой популяции векторов для нормальной работы метода, необходимо 10*количество MV"""

    def _main_calc_func(self, func: Callable[[np.ndarray], np.ndarray]): 
        self.model.fit(self.history_to_opt_model_data,self.res_history_to_opt_model_data)
        next_x = self._propose_location()
        self.history_to_opt_model_data = np.vstack([self.history_to_opt_model_data, next_x.copy()])

        next_x_for_fun = next_x.copy()
        for idx in self.discrete_indices:
            next_x_for_fun[idx] = 1 if next_x_for_fun[idx]>=0.5 else 0
        output_value = func(next_x_for_fun)
        self.res_history_to_opt_model_data.append(output_value)
        if self.target_to_opt:
            self.res_of_most_opt_vec = max(output_value[0], self.res_of_most_opt_vec)
        else:
            self.res_of_most_opt_vec = min(output_value[0], self.res_of_most_opt_vec)
        
        self.most_opt_vec = self.history_to_opt_model_data[np.array(self.res_history_to_opt_model_data)[:,0].tolist().index(self.res_of_most_opt_vec)]
        
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
        self.input_bound_of_vec = self._bound_func_("to_model")
        self.output_bound_of_vec = self._bound_func_("from_model")

        discrete_indices = []
        for i, prop in enumerate(self._to_opt_model_data._values_properties_list):
            if isinstance(prop, _boolItem):
                discrete_indices.append(i)
        
        # Добавляем явно указанные дискретные индексы
        discrete_indices.extend(self.discrete_indices)
        self.discrete_indices = list(set(discrete_indices))  # Удаляем дубликаты

        
        self.history_to_opt_model_data = self._init_vecs(20)
        #print(self.history_to_opt_model_data)
        history_for_fun = self.history_to_opt_model_data
        for vec in history_for_fun:
            for idx in self.discrete_indices:
                vec[idx] = 1 if vec[idx]>=0.5 else 0

        res_list = []

        for vec in history_for_fun:
            output_value = func(vec)
            res_list.append(output_value)

        self.res_history_to_opt_model_data = res_list

        if self.target_to_opt:
            self.res_of_most_opt_vec = max(np.array(res_list)[:,0])
        else:
            self.res_of_most_opt_vec = min(np.array(res_list)[:,0])

        self.most_opt_vec = self.history_to_opt_model_data[np.array(self.res_history_to_opt_model_data)[:,0].tolist().index(self.res_of_most_opt_vec)]

        for _ in range(self._iteration_limitation):
            self.input_bound_of_vec = self._bound_func_("to_model")
            self.output_bound_of_vec = self._bound_func_("from_model")
            self._main_calc_func(func=func)
            
    """Функция инициализации и оптимизации"""

    def getResult(self):
        result = self.most_opt_vec
        for idx in self.discrete_indices:
            result[idx] = 1 if result[idx]>=0.5 else 0
        return result
    """Функция результата, возращает точку"""

    def get_y(self):
        result = self.most_opt_vec
        i = np.where((self.history_to_opt_model_data == result).all(axis=1))[0][0]
        return self.res_history_to_opt_model_data[i]