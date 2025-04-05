"""
main.py

Файл отладки библиотеки

NOTE:
>>> python -m pipreqs.pipreqs .


"""


# Импорт основного класса оптимизации
from BlackBoxOptimizer import Optimizer
from BlackBoxOptimizer import TestShaffleOpt



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




if __name__ == "__main__":

    # Создать класс оптимизатора
    opt = Optimizer(
        optCls              = TestShaffleOpt,
        seed                = 1546, 
        to_model_vec_size   = 21,
        from_model_vec_size = 3,
        iter_limit          = 10
        )

    # Пример конфигурирования для конктретной реализации оптимизирущего класса
    opt.configure(seed = 24657)

    # Запуск оптимизации
    opt.modelOptimize(func = test_object_function_variant_B)

    # Пример запроса истории изменения вектора, отправляемого в модель
    print(opt.getHistoricalData("vec_to_model"))