from ..BaseInterface import BaseExternalModelInterface

import numpy as np

class InternalTestInterface(BaseExternalModelInterface):
    """
    InternalTestInterface
    ---
    Реализация взаимодействия с внешней моделью для внутреннего тестирования методов оптимизации.
    В данном случае внешняя модель - собственные написанные функции с известной оптимальной точкой

    """

    def __init__(self, *args, **kwargs):
        """
        __init__
        ---
        Конструктор класса интерфейса взаимодействия с внешней моделью для внутреннего тестирования методов оптимизации.
        Учитывает запись числа обращений для оценки методов оптимизации

        """

        super().__init__(*args, **kwargs)
        self._usage_count: int = 0
        """Число обращений к внешней модели"""

    def evaluate_external_model(self, to_vec: np.ndarray) -> np.ndarray:
        """
        evaluate_external_model
        ---
        Реализация для внутреннего тестирования. Увеличивает счётчик числа вызовов внешней модели.

        """

        self._usage_count += 1
        return self._external_model.evaluate(to_vec)

    def get_usage_count(self) -> int:
        """
        get_usage_count
        ---
        Возвращает число вызовов внешней модели

        """

        return self._usage_count

    def refresh_usage_count(self) -> None:
        """
        refresh_usage_count
        ---
        Обнуляет число вызовов внешней модели

        """

        self._usage_count = 0

    def set_model(self, external_model) -> None:
        """
        set_model
        ---
        Устанавливает объект внешней модели

        """
        self._usage_count = 0
        """При смене модели обязательно обнулим число обращений к ней"""
        self._external_model = external_model

    def get_true_opimum(self) -> np.ndarray:
        """
        get_true_opimum
        ---
        Возвращает истинное оптимальное значение для данной модели

        """
        if true_optimum := self._external_model.true_optimum is None:
            true_optimum = self._external_model.calculate_true_optimum()

        return true_optimum
