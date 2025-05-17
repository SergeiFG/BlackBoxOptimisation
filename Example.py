from BlackBoxOptimizer import TestStepOpt, GaussOpt
from BlackBoxOptimizer import Optimizer, OptimisationTypes

import numpy as np

from Models import DiscreteSquareSumModel


import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":


    target_point = np.array([2, 0.5, -0.2, 1, 0])  # Целевая точка, которую хотим увидеть, используется для отладки
    discrete_index = np.array([3, 4])
    model = DiscreteSquareSumModel(target=target_point,discrete_indices=discrete_index)

    # Создать класс оптимизатора
    opt = Optimizer(
        optCls              = GaussOpt,
        seed                = 146, # TODO: Проверить, точно ли работает. Сейчас выдаёт разные значения при одном seed
        to_model_vec_size   = 5,
        from_model_vec_size = 1,
        iter_limit          = 200,
        external_model = model.evaluate,
        # user_function = lambda x: x[0],
        optimisation_type = OptimisationTypes.minimize,
        target = None,
        )

    # Пример конфигурирования для конктретной реализации оптимизирущего класса
    opt.configure(kernel_cfg=('RBF',{}))
    opt.setVecItemType(3, "bool", "to_model")
    opt.setVecItemType(4, "bool", "to_model")
    # Запуск оптимизации
    opt.modelOptimize()
    currentOptimizer = opt.getOptimizer()
    print('История изменения рабочей точки')
    print(*currentOptimizer.history_to_opt_model_data)
    # print(20*'=')
    # print('История вычисления внешней моделью черным ящиком')
    # print(currentOptimizer.history_from_model_data)
    print(20 * '=')
    print('Результат')
    print(currentOptimizer.getResult())




