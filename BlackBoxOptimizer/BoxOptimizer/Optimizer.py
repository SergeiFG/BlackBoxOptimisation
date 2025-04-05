"""@package docstring
Исхоныдй класс оптимизатора

More details.
"""
from .BaseOptimizer import BaseOptimizer

import numpy as np
from typing import Callable, Literal



class Optimizer(object):
    """
    Optimizer
    ---
    Функция оптимизации черного ящика
    
    """
    def __init__(self, 
                 optCls : object = None,
                 *args, 
                 **kwargs
                 ) -> None:
        """
        __init__
        ---
        Аргументы:
            optCls : object - выбранный класс оптимизации
        """

        self._currentOptimizerClass = optCls
        """Текущий оптимизирующий класс"""

        if self._currentOptimizerClass is None:
            raise ValueError("Не указан оптимизируйщий класс")

        self._CurrentOptimizerObject = self._currentOptimizerClass(*args, **kwargs)
        """Текущий объект оптимизации"""



    def configure(self, **kwargs) -> None:
        """
        configure
        ---
        Метод обновления конфигурации
        
        Выполняет конфигурирование параметров работы алгоритмов при наличии необходимых
        
        Пример использования:
        
        >>> import SpecificOptimizer
        ... optimizer = SpecificOptimizer()
        ... optimizer.configure(some_parameter_A = some_value_A, some_parameter_B = some_value_B)
        """
        return self._CurrentOptimizerObject.configure(**kwargs)



    def modelOptimize(self, func : Callable[[np.array], np.array]) -> None:
        """
        modelOptimize
        ---
        Запуск оптимизации через передачу функции черного ящика
        """
        return self._CurrentOptimizerObject.modelOptimize(func)



    def getResult(self) -> np.ndarray:
        """
        getResult
        ---
        Возврат результата работы модуля оптимизации черного ящика
        """
        return self._CurrentOptimizerObject.getResult()



    def getHistoricalData(self, key : None | Literal["vec_to_model", "vec_from_model", "obj_val"] = None) -> None | list:
        """
        getHistoricalData
        ---
        Получение исторических данных, собранных в результате работы алгоритма
        
        "vec_to_model" - Получение истории векторов, отправляемых в оптимизационную модель на протя-
                         жении всех заданных итераций работы
        "vec_from_model" - Получение истории векторов, получаемых от оптимизационной модели на протя-
                           жении всех заданных итераций работы
        "obj_val" - Зафиксированные значения целевой функции
        """
        return self._CurrentOptimizerObject.getHistoricalData(key)
    
    