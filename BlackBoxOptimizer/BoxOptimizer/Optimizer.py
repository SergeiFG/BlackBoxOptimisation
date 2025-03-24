"""@package docstring
Исхоныдй класс оптимизатора

More details.
"""
from .BaseOptimizer import BaseOptimizer

import numpy as np


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



# TODO: Продумать архитекутру обращения к модели оптимизации. Пока простой доступ через 
#       относительно защищенные атрибуты

    def AlgIter(self):
        """
        AlgIter 
        ---
        Итератор работы с алгоритмом 
        
        """
        return self._CurrentOptimizerObject.AlgIter()

    # Пока просто ретранслируем на класс выше, позже подумать в рамках архитектурной реализации обра-
    # щения к модели
    @property
    def vecToModel(self) -> np.array:
        """Вектор, оправляемый в модель оптимизации"""
        return self._CurrentOptimizerObject.vecToModel

    @property
    def vecFromModel(self) -> None:
        # raise AttributeError("Чтение атрибута не допускается")
        return self._CurrentOptimizerObject.vecFromModel

    @vecFromModel.setter
    def vecFromModel(self, new_value : np.array) -> None:
        """Установка значений, полученных от модели"""
        self._CurrentOptimizerObject.vecFromModel = new_value

    @property
    def objFuncValue(self) -> float | None:
        """
        Чтиение установленного значения целевой функции
        """
        return self._CurrentOptimizerObject.objFuncValue
    
    @objFuncValue.setter
    def objFuncValue(self, new_val : float | int | None) -> None:
        """
        Установка занчения целевой функции от модели
        """
        self._CurrentOptimizerObject.objFuncValue = new_val