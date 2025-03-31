"""
Тестовый оптимизатор для проверки и демонстрации поиска оптимальной точки

Тестовая реализация алгоритма оптимизации путем координатного поиска с фиксированным шагом
"""

from ..BaseOptimizer import BaseOptimizer

import numpy as np

class TestStepOpt(BaseOptimizer):
    """
    TestStepOpt
    ---
    Тестовая реализация алгоритма оптимизации путем координатного поиска с фиксированным шагом

    """
    def __init__(self, seed: int, step: float | int = 1.0, *args, **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса оптимизатора
        """
        super().__init__(*args, **kwargs)

        self.seed: int = seed
        """База генератора рандомных чисел"""
        self.step: float | int = step
        """Фиксированный шаг для расчёта"""
        self.history_to_opt_model_data = []
        self.history_from_model_data = []



    def _main_calc_func(self):
        """
        _main_calc_func
        ---
        Координатный поиск с фиксированным шагом
        """
        history_to = []
        history_from = []
        for to_vec, from_vec in zip(self._to_opt_model_data.iterVectors(), self._from_model_data.iterVectors()):
            """Ниже Данные для записи в историю точка в начале каждой итерации"""
            history_to.append(to_vec.copy())

            """Ниже основной цикл работы модели оптимизации"""
            to_vec[:] = to_vec[:] + self.step

            """Ниже Данные для записи в историю итоги из внешней модели"""
            history_from.append(from_vec.copy())

        self.history_to_opt_model_data.append(history_to.copy())
        self.history_from_model_data.append(history_from.copy())

    def getResult(self):
        return self._to_opt_model_data.iterVectors()