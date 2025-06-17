from BlackBoxOptimizer import TestStepOpt, GaussOpt
from BlackBoxOptimizer import Optimizer, OptimisationTypes

import numpy as np

from Models import SquareSumModel


import warnings

warnings.filterwarnings('ignore')


if __name__ == "__main__":


    target_point = np.array([0.5, -0.5, 0, 0])  # Целевая точка, которую хотим увидеть, используется для отладки
    # discrete_index = np.array([4, 5])
    model = SquareSumModel(target=target_point)

    #Создать класс оптимизатора
    opt = Optimizer(
        optCls              = GaussOpt,
        seed                = 146, # TODO: Проверить, точно ли работает. Сейчас выдаёт разные значения при одном seed
        to_model_vec_size   = 4,
        from_model_vec_size = 1,
        iter_limit          = 200,
        external_model = model.evaluate,
        # user_function = lambda x: x[0],
        optimisation_type = OptimisationTypes.minimize,
        target = None,
        )

    # Пример конфигурирования для конктретной реализации оптимизирущего класса
    opt.configure(kernel_cfg=('RBF',{}))
    # opt.setVecItemType(4, "bool", "to_model")
    # opt.setVecItemType(5, "bool", "to_model")
    for i in range(4):
        opt.setVecItemLimit(i, "to_model", -10, 10)
    # for i in range(4):
    #     opt.setVecItemLimit(i, "from_model", -1, 1.9)
    # Запуск оптимизации
    # opt.setVecItemLimit(1, "from_model", min=1, max=np.inf)
    opt.modelOptimize()
    currentOptimizer = opt.getOptimizer()
    print('История изменения рабочей точки')
    print(*currentOptimizer.history_to_opt_model_data)
    print(20*'=')
    print('История вычисления внешней моделью черным ящиком')
    print(*currentOptimizer.res_history_to_opt_model_data)
    print(20 * '=')
    print('Результат')
    print(currentOptimizer.getResult())
    print(model.evaluate(currentOptimizer.getResult()))
    print(currentOptimizer.get_y())