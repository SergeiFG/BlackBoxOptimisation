from BlackBoxOptimizer import TestStepOpt
from BlackBoxOptimizer import Optimizer

import numpy as np
from typing import Tuple


class BaseExternalModel:
    def __init__(self):
        self._call_count = 0

    def _increment_call_count(self):
        """Увеличивает счетчик вызовов метода evaluate."""
        self._call_count += 1

    def evaluate(self, to_vec: np.ndarray, *args, **kwargs):
        """Метод-заглушка для переопределения в наследниках."""
        raise NotImplementedError("Метод evaluate должен быть переопределен в подклассе")

    def _track_evaluate_calls(func):
        """Декоратор для отслеживания числа вызовов evaluate."""
        def wrapper(self, vec):
            self._increment_call_count()
            return func(self, vec)
        return wrapper

    def get_call_count(self):
        return self._call_count


class ModelMinSquareSum(BaseExternalModel):
    def __init__(self, target: np.ndarray):
        super().__init__()
        self.target = target

    @BaseExternalModel._track_evaluate_calls
    def evaluate(self, to_vec, *args, **kwargs):
        return np.array([np.sum((to_vec - self.target) ** 2), max(to_vec)])



if __name__ == "__main__":

    # Создать класс оптимизатора
    opt = Optimizer(
        optCls              = TestStepOpt,
        seed                = 1546, # TODO: Проверить, точно ли работает. Сейчас выдаёт разные значения при одном seed
        to_model_vec_size   = 3,
        from_model_vec_size = 2,
        iter_limit          = 100
        )

    # Пример конфигурирования для конктретной реализации оптимизирущего класса
    opt.configure(step = 0.01)

    target_point = np.array([0, 0.5, -0.2]) # Целевая точка, которую хотим увидеть, используется для отладки
    model = ModelMinSquareSum(target_point)

    # Запуск оптимизации
    opt.modelOptimize(func = model.evaluate)
    currentOptimizer = opt.getOptimizer()
    print('История изменения рабочей точки')
    print(*currentOptimizer.history_to_opt_model_data)
    print(20*'=')
    print('История вычисления внешней моделью черным ящиком')
    print(currentOptimizer.history_from_model_data)
    print(20 * '=')
    print(f'Число вызовов внешней модели - {model.get_call_count()}')
    print(20 * '=')
    print('Результат')
    print(currentOptimizer.getResult())



