from BlackBoxOptimizer import TestStepOpt
from BlackBoxOptimizer import Optimizer

import numpy as np
from typing import Tuple



def test_object_function(values : np.array) -> Tuple[np.array, float]:
    """
    Тестовая целевая функция с выдачей какого-то вектора
    Возврат:
        - Тот самый какой-то вектор
        - Значение целевой функции
    """
    return (values.copy()[0:10] , np.sum(values.copy().sum()) * 4)


def test_object_function_variant_B(values : np.array) -> np.array:
    """
    Тестовая целевая функция с выдачей какого-то вектора
    Возврат:
        - Тот самый какой-то вектор
    """
    loc_vec = values.copy()[0:3]
    loc_vec[0] = np.sum(values.copy().sum()) * 4
    return loc_vec.copy()

class external_model:
    def __init__(self, param):
        self.parameter = param
        self.usage_count = 0

    def evaluate(self, to_vec):
        self.usage_count += 1
        self.parameter += 1
        return np.array([self.parameter, max(to_vec)])



if __name__ == "__main__":

    # Создать класс оптимизатора
    opt = Optimizer(
        optCls              = TestStepOpt,
        seed                = 1546,
        to_model_vec_size   = 3,
        from_model_vec_size = 2,
        iter_limit          = 3
        )

    # Пример конфигурирования для конктретной реализации оптимизирущего класса
    opt.configure(step = 0.5)

    model = external_model(5)

    # Запуск оптимизации
    opt.modelOptimize(func = model.evaluate)
    currentOptimizer = opt.getOptimizer()
    print(*currentOptimizer.history_to_opt_model_data)
    print(20*'=')
    print(currentOptimizer.history_from_model_data)
    print(20 * '=')
    print(model.usage_count)

