from ..BaseModel import BaseModel

import numpy as np
from typing import Tuple


class ConstrainedParabolaModel(BaseModel):
    """
    ParabolaModel
    ---
    Класс тестирования внешней модели, который реализует вычисление функции с ограничениями на x и f(x)
    Для определённости работает исключительно с минимизацией
    f(x) = a(x+b)**2 + c
    с оптимальной точкой x = b | min_x | max_x
    """
    def __init__(self, a: float, b: float, c: float,
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
        self.b = b
        self.func = lambda x: float(a * (x + b)**2 + c)
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
        candidates = np.array([self.constrains_x[0], self.b, self.constrains_x[1]])
        tmp = [self.func(x) for x in candidates]
        best_ind = np.argmin(tmp)
        if (self.constrains_y is None) or (self.constrains_y[0] < tmp[best_ind]):
            return candidates[best_ind]
        else:
            return 'Минимум в непонятной точке'
        """ Оптимальная точка в экстремуме или на границе если значение входит в границы по f(x)
        Считаем, что значение в одной из этих точек меньше верхней границы, иначе задача недостижима
        Иначе лучшее значение при f(x) == min_y, мне лениво этот вариант рассматривать"""




