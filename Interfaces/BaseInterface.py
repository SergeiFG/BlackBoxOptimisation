import numpy as np

from typing import Callable, Tuple
import enum


class OptimisationTypes(enum.Enum):
    minimize = 0
    maximize = 1
    to_target = 2


optimization_transform_funcs = {
    OptimisationTypes.minimize: lambda x, y = None: x,
    OptimisationTypes.maximize: lambda x, y = None: -x,
    OptimisationTypes.to_target: lambda x, y: x - y,
}
"""Словарь преобразования задачи оптимизации для приведения её к виду argmin(f(x))"""


class BaseExternalModelInterface:
    def __init__(self,
                 external_model: Callable[[np.ndarray], np.ndarray], # TODO: Подумать, это должен быть класс или просто функция
                 user_function: Callable[[np.ndarray], float] = lambda x: x[0],
                 optimisation_type: OptimisationTypes = OptimisationTypes.minimize,
                 target: float | None = None
                 ) -> None:
        """
        __init__
        ---
        Конструктор интерфейса для взаимодействия с внешней моделью - черным ящиком.
        Применяется для обеспечения взаимодействия внешних моделей с модулем оптимизации

        Аргументы:
            external_model - Функция вычисления внешней модели
            user_function: Callable[[np.ndarray], float] - Пользовательская функция, применяемая к выходу внешней модели для формирования задачи оптимизации
            optimisation_type: OptimisationTypes - Тип задачи с целевой функцией оптимизации [минимизировать, максимизировать, приблизить к указанному значению], применяется к результату пользовательской функции
            target  : float - Целевое значение целевой функции при выборе задачи OptimisationTypes.to_target, применяется к результату пользовательской функции

        """

        self.external_model = external_model
        self.user_function = user_function
        self.optimisation_type = optimisation_type
        self.target = target

    def configure(self, **kwargs) -> None:
        """
        configure
        ---
        Метод настройки параметров работы интерфейса взаимодействия с внешней моделью
        """
        for key, value in kwargs.items():

            # Проверка наличия атрибута настройки параметров работы интерфейса взаимодействия с внешней моделью
            if key not in self.__dict__:
                raise KeyError(f"{self.__class__.__name__} не содержит настраиваемого параметра {key}")
            else:
                self.__dict__[key] = value

    def evaluate_external_model(self, to_vec: np.ndarray) -> np.ndarray:
        """
        evaluate_external_model
        ---
        Метод обращения к внешней модели
        """

        return self.external_model(to_vec)

    def apply_user_func(self, from_vec: np.ndarray) -> float:
        """
        apply_user_func
        ---
        Метод применения пользовательской функции к выходу внешней модели
        """

        return self.user_function(from_vec)

    def transform_optimisation_type(self, value: float) -> float:
        """
        transform_optimisation_type
        ---
        Метод преобразования выхода пользовательской функции для приведения задачи к минимизации
        """

        if self.optimisation_type is OptimisationTypes.to_target and self.target is None:
            raise AttributeError('Не задана цель оптимизации')

        transform_func = optimization_transform_funcs[self.optimisation_type]
        return transform_func(value, self.target)

    def evaluate(self, to_vec: list[np.ndarray]) -> list[Tuple[float, np.ndarray]]:
        """
        evaluate
        ---
        Метод вычисления внешней модели.
        Вход - набор кандидатов MV - массив, каждый элемент которого потенциальный кандидат вектор принимаемых внешней моделью MV
        Выход - Набор результатов вычисления для каждого кандидата - массив, каждый элемент которого кортеж (целевая переменная, numpy массив возвращаемых CV)
        """

        res = []
        for candidate in to_vec:
            from_vec = self.evaluate_external_model(candidate)
            user_val = self.apply_user_func(from_vec)
            optimisation_value = self.transform_optimisation_type(user_val)
            res.append((optimisation_value, from_vec.copy()))

        return res





