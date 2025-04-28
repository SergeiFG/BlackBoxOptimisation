from BlackBoxOptimizer import Optimizer, OptimisationTypes
from Models import SquareSumModel
import numpy as np
from BlackBoxOptimizer import SimulatedAnnealingOptimizer

if __name__ == "__main__":
    target_point = np.array([0, 0.5, -0.2])
    model = SquareSumModel(-target_point)

    opt = Optimizer(
        optCls=SimulatedAnnealingOptimizer,
        seed=1546,
        to_model_vec_size=3,
        from_model_vec_size=1,
        iter_limit=1000,
        external_model=model.evaluate,
        optimisation_type=OptimisationTypes.minimize,
        initial_temperature=100.0,
        cooling_rate=0.99,
        temperature_min=0.1
    )


    opt.setVecItemLimit(0, min=-10.0, max=10.0)  
    opt.setVecItemLimit(1, min=-10.0, max=10.0)  
    opt.setVecItemLimit(2, min=-10.0, max=10.0) 


    opt.modelOptimize()
    
    result = opt.getResult()
    print('\nРезультат оптимизации:')
    print(f"Найденная точка: {result[0]}")
    print(f"Значение функции: {result[1]:.6f}")
    print(f"Истинный оптимум: {target_point}")
    print(f"Расстояние до оптимума: {np.linalg.norm(result[0] - target_point):.6f}")
    print(f"Число итераций: {opt.get_usage_count()}")