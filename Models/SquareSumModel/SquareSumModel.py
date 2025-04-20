from ..BaseModel import BaseModel

import numpy as np


class SquareSumModel(BaseModel):
    """
    SquareSumModel
    ---
    Класс тестирования внешней модели, который реализует вычисление функции без ограничений
    f(x) = (x1 + a)**2 + (x2 + b)**2 + (x3 + c)**2 +...
    с оптимальной точкой x = [a, b, c, ..]
    """
    def __init__(self, target: np.ndarray, **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса внешней модели параболы, используемой для тестирования алгоритмов оптимизации
        """

        super().__init__(**kwargs)

        self.true_optimum = -target
        self.func = lambda x: np.sum((x + target) ** 2)

    def evaluate(self, to_vec: np.ndarray) -> np.ndarray:
        """
        evaluate
        ---
        Метод вычисления CV внешней модели
        """
        return np.array([self.func(to_vec)])

    def calculate_true_optimum(self) -> np.ndarray:
        """
        calculate_true_optimum
        ---
        Метод вычисления истинно оптимального значения
        """
        pass

