from Models import SquareSumModel
from BlackBoxOptimizer import GaussOpt
from BlackBoxOptimizer import Optimizer, OptimisationTypes

import numpy as np

if __name__ == "__main__":


    target_point = np.array([-0.8, 1.7])  # Целевая точка, которую хотим увидеть, используется для отладки

    model = SquareSumModel(target=-target_point)

    opt = Optimizer(
        optCls              = GaussOpt,
        seed                = 146, # TODO: Проверить, точно ли работает. Сейчас выдаёт разные значения при одном seed
        to_model_vec_size   = 2,
        from_model_vec_size = 1,
        iter_limit          = 80,
        external_model = model.evaluate,
        optimisation_type = OptimisationTypes.minimize,
        target = None,
        )
    for j in range(2):
        opt.setVecItemLimit(j, "to_model", -3, 3)
    opt.modelOptimize()
    currentopt = opt.getOptimizer()
    print('История изменения рабочей точки')
    print(*currentopt.history_to_opt_model_data)
    print(20*'=')
    print(*currentopt.X_scaled)
    print(20*'=')
    print(opt.getResult())