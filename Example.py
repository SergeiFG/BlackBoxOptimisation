from BlackBoxOptimizer import TestStepOpt
from BlackBoxOptimizer import Optimizer, OptimisationTypes

import numpy as np

from Models import SquareSumModel


if __name__ == "__main__":


    target_point = np.array([0, 0.5, -0.2])  # Целевая точка, которую хотим увидеть, используется для отладки
    model = SquareSumModel(-target_point)

    # Создать класс оптимизатора
    opt = Optimizer(
        optCls              = TestStepOpt,
        seed                = 1546, # TODO: Проверить, точно ли работает. Сейчас выдаёт разные значения при одном seed
        to_model_vec_size   = 3,
        from_model_vec_size = 2,
        iter_limit          = 100,
        external_model = model.evaluate,
        # user_function = lambda x: x[0],
        optimisation_type = OptimisationTypes.minimize,
        target = None
        )

    # Пример конфигурирования для конктретной реализации оптимизирущего класса
    opt.configure(step = 0.01, user_function = lambda x: x[0])

    # Запуск оптимизации
    opt.modelOptimize()
    currentOptimizer = opt.getOptimizer()
    print('История изменения рабочей точки')
    print(*currentOptimizer.history_to_opt_model_data)
    print(20*'=')
    print('История вычисления внешней моделью черным ящиком')
    print(currentOptimizer.history_from_model_data)
    print(20 * '=')
    print(f'Число вызовов внешней модели - {opt.get_usage_count()}')
    print(20 * '=')
    print('Результат')
    print(currentOptimizer.getResult())



