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




if __name__ == "__main__":

    # Создать класс оптимизатора 
    opt = Optimizer(
        optCls = TestShaffleOpt,
        seed = 1546, 
        to_model_vec_size = 21,
        from_model_vec_size = 10,
        iter_limit = 10
        )

    # Пример конфигурирования для конктретной реализации оптимизирущего класса
    opt.configure(seed = 24657)


    # Пример запуска функционала оптимизации. Временный.
    # Имеет ряд недостатков, связанный с возрастающим числом ошибок использования функционала
    for _ in range(15):
        loc_val : Tuple[np.array, float] = test_object_function(opt.vecToModel)
        opt.vecFromModel = loc_val[0]
        opt.objFuncValue = loc_val[1]





    # TODO: Оставлено до следующей отладки
    # for _ in range(15):
        
    #     print("BEFORE opt.vecFromModel", opt.vecFromModel)
    #     print("BEFORE opt.vecToModel", opt.vecToModel)
    #     loc_val : Tuple[np.array, float] = test_object_function(opt.vecToModel)
 
    #     opt.vecFromModel = loc_val[0]
    #     print("AFTER opt.vecFromModel", opt.vecFromModel)
    #     # print(opt.vecFromModel)
    #     opt.objFuncValue = loc_val[1]
    #     print(opt.objFuncValue)
    #     print("\n")




    # TODO: Необходим пересмотр Итератора
    # Итерируемся, передавая промежуточные значения вектора в модель
    # loc_val : Tuple[np.array, np.array]
    # for to_model_vec, from_model_vec, obj_func_val in opt.AlgIter():
    #     loc_val = test_object_function(to_model_vec)
    #     print(type(loc_val[0])) 
    #     print("from_model_vec", type(from_model_vec))
    #     print("from_model_vec[:]", type(from_model_vec[:]))

    #     from_model_vec[:] = loc_val[0]
    #     obj_func_val[:]   = loc_val[1]


