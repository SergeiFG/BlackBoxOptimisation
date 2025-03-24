"""
Тестовый оптимизатор

Тестовая реализация алгоритма оптимизации путем перемешивания значений массива
"""

from ..BaseOptimizer import BaseOptimizer

import numpy as np

class TestShaffleOpt(BaseOptimizer):
    """
    TestShaffleOpt
    ---
    Тестовая реализация алгоритма оптимизации путем перемешивания значений массива
    
    """
    def __init__(self, seed : int, *args, **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса оптимизатора
        """
        super().__init__(*args, **kwargs)
        
        self.seed : int = seed
        """База генератора рандомных чисел"""


    def _main_calc_func(self):
        """
        _main_calc_func
        ---
        Простейший случай главной работы - перемешивание элементов массива и прибавление максимума 
        от оптимизуемой модели
        """
        self._to_opt_model_data.vec = self._to_opt_model_data.vec + np.max(self._from_model_data.vec)
        np.random.shuffle(self._to_opt_model_data.vec)

