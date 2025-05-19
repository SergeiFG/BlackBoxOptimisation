"""
Тестовый оптимизатор для проверки и демонстрации поиска оптимальной точки

Тестовая реализация алгоритма оптимизации путем координатного поиска с фиксированным шагом
"""

from ..BaseOptimizer import BaseOptimizer

import numpy as np
from typing import Callable

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

    def modelOptimize(self, func: Callable[[np.array], np.array]) -> None:
        """
        modelOptimize
        ---
        Запуск оптимизации через передачу функции черного ящика
        Реализация при необходимости вызова внешней модели внутри итерации оптимизации
        Если внутри итерации оптимизации нет вызова внешней модели, оставить базовую реализацию
        """
        for _ in range(self._iteration_limitation):
            self._main_calc_func(func)

    def _main_calc_func(self, func):
        """
        _main_calc_func
        ---
        Координатный поиск с фиксированным шагом
        """
        history_to = []
        history_from = []
        for to_vec, from_vec in zip(self._to_opt_model_data.iterVectors(), self._from_model_data.iterVectors()):
            point = to_vec.copy() # Текущая точка (вектор), будем двигаться последовательно по каждой координате
            for i in range(len(point)): # Генерация для каждого парамета 3 кандидатов - ничего не измменилось, прибавить шаг, отнять шаг
                candidate_low = point.copy()
                candidate_low[i] -= self.step

                candidate_stay = point.copy()

                candidate_high = point.copy()
                candidate_high[i] += self.step

                candidates = [candidate_low, candidate_stay, candidate_high]
                target_values = [func(candidate_low)[0], func(candidate_stay)[0], func(candidate_high)[0]]
                # target_values = func(candidates)
                # target_values = [x[0] for x in target_values]
                """Два варианта обращения к внешней модели, оба работают"""
                '''Вычисление внешней модели для каждого кандидата, по умолчанию сделано что целевая функция - первый элемент массива'''
                
                if hasattr(self, "optimisation_type") and getattr(self, "optimisation_type", None) is not None:
                    if getattr(self, "optimisation_type") == 1 or getattr(self, "optimisation_type").name == "maximize":
                        best_idx = int(np.argmax(target_values))
                    else:
                        best_idx = int(np.argmin(target_values))
                else:
                    best_idx = int(np.argmin(target_values))

                point = candidates[best_idx].copy()
                """Основной цикл работы модели оптимизации"""

            to_vec[:] = point.copy() # Записываем итоговую точку
            history_to.append(to_vec.copy())
            """Данные для записи в историю новая рабочая точка (MV) после итерации"""

            from_vec = func(to_vec)
            history_from.append(from_vec)
            """Данные для записи в историю итоги из внешней модели (CV) после итерации"""

        self.history_to_opt_model_data.append(history_to.copy())
        self.history_from_model_data.append(history_from.copy())

    def getResult(self):
        return list(self._to_opt_model_data.iterVectors())