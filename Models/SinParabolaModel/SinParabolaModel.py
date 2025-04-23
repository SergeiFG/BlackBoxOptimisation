from ..BaseModel import BaseModel

import numpy as np
from math import sin
from typing import Tuple


class SinParabolaModel(BaseModel):
    """
    ParabolaModel
    ---
    Класс тестирования внешней модели, который реализует вычисление функции с ограничениями на x и f(x)
    f(x) = a(x+b)**2 + c + d * sin(e*x + f)
    с оптимальной точкой нужно смотреть по графику
    """
    def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float,
                 constrains_x: Tuple[float, float], constrains_y: Tuple[float, float] | None,
                 **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса внешней модели параболы, используемой для тестирования алгоритмов оптимизации
        """

        super().__init__(**kwargs)

        self.true_optimum = None
        self.constrains_x = constrains_x
        self.constrains_y = constrains_y
        self.func = lambda x: float(a*(x + b)**2 + c + d*sin(e*x + f))
        """Ограничения никак не влияют на вычисление модели и учитываются только алгоритмом оптимизации"""

    def evaluate(self, to_vec: np.ndarray) -> np.ndarray:
        """
        evaluate
        ---
        Метод вычисления CV внешней модели
        """
        return np.array([self.func(to_vec[0])])

    def calculate_true_optimum(self) -> np.ndarray:
        """
        calculate_true_optimum
        ---
        Метод вычисления истинно оптимального значения
        """
        pass # TODO: Сделать реализацию через scipy.optimize




