from ..BaseModel import BaseModel

import numpy as np


class ParabolaModel(BaseModel):
    """
    ParabolaModel
    ---
    Класс тестирования внешней модели, который реализует вычисление функции без ограничений
    f(x) = a(x+b)**2 + c
    с оптимальной точкой x = b при f(x) = c
    """
    def __init__(self, a: float, b: float, c: float, **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса внешней модели параболы, используемой для тестирования алгоритмов оптимизации
        """

        super().__init__(**kwargs)

        self.true_optimum = np.array([float(-b)])
        self.func = lambda x: float(a * (x + b)**2 + c)

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
        pass

