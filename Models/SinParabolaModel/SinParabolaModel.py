from ..BaseModel import BaseModel

import numpy as np
import math
from typing import Tuple


class SinParabolaModel(BaseModel):
    """
    ParabolaModel
    ---
    Класс тестирования внешней модели, который реализует вычисление функции с ограничениями на x и f(x)
    f(x) = a(x+b)**2 + c + d * sin(e*x + f)
    с оптимальной точкой нужно смотреть по графику
    """
    def __init__(self, a, b: float, c: float,
                 **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса внешней модели параболы, используемой для тестирования алгоритмов оптимизации
        """

        super().__init__(**kwargs)

        self.true_optimum = None
        self.func = lambda x: float((x[0] + a[0])**2 + (x[1] + a[1])**2 - b*math.cos(c*x[0] + a[0]*c) - b*math.cos(c*x[1] + a[1]*c)) + 2*b
        """Ограничения никак не влияют на вычисление модели и учитываются только алгоритмом оптимизации"""

    def evaluate(self, to_vec: np.ndarray) -> np.ndarray:
        """
        evaluate
        ---
        Метод вычисления CV внешней модели
        """
        return np.array([self.func(to_vec[:2])])

    def calculate_true_optimum(self) -> np.ndarray:
        """
        calculate_true_optimum
        ---
        Метод вычисления истинно оптимального значения
        """
        pass # TODO: Сделать реализацию через scipy.optimize




