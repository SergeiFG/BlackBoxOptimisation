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



    def _main_calc_func(self, func):
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
            point = to_vec.copy() # Текущая точка (вектор), будем двигаться последовательно по каждой координате
            for i in range(len(point)): # Генерация для каждого парамета 3 кандидатов - ничего не измменилось, прибавить шаг, отнять шаг
                candidate_low = point.copy()
                candidate_low[i] -= self.step

                candidate_stay = point.copy()

                candidate_high = point.copy()
                candidate_high[i] += self.step

                candidates = [candidate_low, candidate_stay, candidate_high]
                '''Вычисление внешней модели для каждого кандидата, по умолчанию сделано что целевая функция - первый элемент массива'''
                target_values = [func(candidate_low)[0], func(candidate_stay)[0], func(candidate_high)[0]]
                point = candidates[np.argmin(target_values)].copy() # Обновляем точку, записываем в неё лучшего из кандидатов

            to_vec[:] = point.copy() # Записываем итоговую точку

            """Ниже Данные для записи в историю итоги из внешней модели"""
            history_from.append(from_vec.copy())

        self.history_to_opt_model_data.append(history_to.copy())
        self.history_from_model_data.append(history_from.copy())

    def getResult(self):

        return list(self._to_opt_model_data.iterVectors())