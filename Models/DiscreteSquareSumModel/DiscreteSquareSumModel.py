from ..BaseModel import BaseModel

from typing import List

import numpy as np


class DiscreteSquareSumModel(BaseModel):
    """
    SquareSumModel
    ---
    Класс тестирования внешней модели, который реализует вычисление функции без ограничений
    f(x) = (x1 + a)**2 + (x2 + b)**2 + (x3 + c)**2 +...
    с оптимальной точкой x = [a, b, c, ..]
    В случае неправильно заданых дискретных параметров точка оптимума сдвигается на 0.1 за каждый неправильный дискретный параметр
    , а минимальное значение фукнции возрастает на 0.1 за каждый неправильный дискретный параметр
    """
    def __init__(self, target: np.ndarray, discrete_indices: List[int] = None, **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса внешней модели параболы, используемой для тестирования алгоритмов оптимизации
        """

        super().__init__(**kwargs)

        self.true_optimum = target
        self.discrete_indices = discrete_indices if discrete_indices is not None else []
        self.func = lambda x: self.function(x)

    def function(self, x):
        mask = np.take(self.true_optimum, self.discrete_indices) == np.take(x,self.discrete_indices)
        return np.sum((np.delete(x,self.discrete_indices)-np.delete(self.true_optimum,self.discrete_indices) - (len(mask) - np.sum(mask))*0.1)**2)+0.1*(len(mask) - np.sum(mask))

    def evaluate(self, to_vec: np.ndarray) -> np.ndarray:
        """
        evaluate
        ---
        Метод вычисления CV внешней модели
        """
        return np.array([self.func(to_vec), to_vec[0]*1, to_vec[1]*4])

    def calculate_true_optimum(self) -> np.ndarray:
        """
        calculate_true_optimum
        ---
        Метод вычисления истинно оптимального значения
        """
        pass

