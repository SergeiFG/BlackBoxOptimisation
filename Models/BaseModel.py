import numpy as np


class BaseModel:
    """
    BaseModel
    ---
    Класс базовой реализации внешней модели.
    Используется ТОЛЬКО для внутреннего тестирования, для интеграции с действительной внешней моделью требуется собственная реализация.
    """

    def __init__(self, **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса внешней модели, используемой для тестирования алгоритмов оптимизации
        """
        for key, value in kwargs.items():
            self.__dict__[key] = value

        self.true_optimum = None

    def evaluate(self, to_vec: np.ndarray) -> np.ndarray:
        """
        evaluate
        ---
        Метод вычисления CV внешней модели
        """
        raise NotImplementedError

    def calculate_true_optimum(self) -> np.ndarray:
        """
        calculate_true_optimum
        ---
        Метод вычисления истинно оптимального значения
        """
        raise NotImplementedError

