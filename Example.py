from BlackBoxOptimizer import TestStepOpt
from BlackBoxOptimizer import Optimizer

import numpy as np

from Interfaces import InternalTestInterface
from Interfaces import OptimisationTypes

from Models import SquareSumModel


if __name__ == "__main__":

    # Создать класс оптимизатора
    opt = Optimizer(
        optCls              = TestStepOpt,
        seed                = 1546, # TODO: Проверить, точно ли работает. Сейчас выдаёт разные значения при одном seed
        to_model_vec_size   = 3,
        from_model_vec_size = 2,
        iter_limit          = 10
        )

    # Пример конфигурирования для конктретной реализации оптимизирущего класса
    opt.configure(step = 0.1)

    target_point = np.array([0, 0.5, -0.2]) # Целевая точка, которую хотим увидеть, используется для отладки
    model = SquareSumModel(-target_point)

    Interface = InternalTestInterface(
        external_model=model,
        user_function=lambda x: x[0],
        optimisation_type=OptimisationTypes.minimize,
        target=None
    )

    # Запуск оптимизации
    opt.modelOptimize(func = Interface.evaluate)
    currentOptimizer = opt.getOptimizer()
    print('История изменения рабочей точки')
    print(*currentOptimizer.history_to_opt_model_data)
    print(20*'=')
    print('История вычисления внешней моделью черным ящиком')
    print(currentOptimizer.history_from_model_data)
    print(20 * '=')
    print(f'Число вызовов внешней модели - {Interface.get_usage_count()}')
    print(20 * '=')
    print('Результат')
    print(currentOptimizer.getResult())



