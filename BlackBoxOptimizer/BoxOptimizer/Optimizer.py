"""@package docstring
Исхоныдй класс оптимизатора

More details.
"""
from .BaseOptimizer import BaseOptimizer

import numpy as np
from typing import Callable



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
        
        
        """
        self._CurrentOptimizerObject.configure(**kwargs)



    def modelOptimize(self, func : Callable[[np.array], np.array]) -> None:
        """
        modelOptimize
        ---
        Запуск оптимизации через передачу функции черного ящика
        """
        self._CurrentOptimizerObject.modelOptimize(func)

    def getOptimizer(self):
        """
        getOptimizer
        ---
        Возврат экземпляр модели оптимизации, с которой он работал
        Необходимо для отладки, получения метрик, тестирования
        """
        return self._CurrentOptimizerObject