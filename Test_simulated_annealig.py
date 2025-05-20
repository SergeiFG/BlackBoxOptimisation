from BlackBoxOptimizer import Optimizer, OptimisationTypes
from Models import SquareSumModel
import numpy as np
from BlackBoxOptimizer import SimulatedAnnealingOptimizer

def test_simulated_annealing():
    target_point = np.array([1, 84, 72, 0, 1, 1])
    dimension = len(target_point)
    model = SquareSumModel(-target_point)
    
    opt = Optimizer(
        optCls=SimulatedAnnealingOptimizer,
        seed=1424,
        to_model_vec_size=6,
        from_model_vec_size=1,
        iter_limit=100000,
        external_model=model.evaluate,
        optimisation_type=OptimisationTypes.minimize,
        initial_temp=50.0,
        min_temp=1e-5,
        cooling_rate=0.98,
        step_size=0.8,
        penalty_coef=1e6  
    )
    
    opt.setVecItemLimit(0, vec_dir="to_model", min=0, max=100)
    opt.setVecItemLimit(1, vec_dir="to_model", min=0, max=100)
    opt.setVecItemLimit(2, vec_dir="to_model", min=0, max=100)
    opt.setVecItemType(3, vec_dir="to_model", new_type="bool")
    opt.setVecItemType(4, vec_dir="to_model", new_type="bool")
    opt.setVecItemType(5, vec_dir="to_model", new_type="bool")
    
    opt.setVecItemLimit(1, vec_dir="from_model", min=2, max=100)


    initial_guess = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    opt.setPreSetCadidateVec(0, initial_guess)
    
    opt.modelOptimize()
    
    result = opt.getResult()
    output_values = model.evaluate(result)
    final_value = output_values[0]
    calls_count = opt.get_usage_count()
    
    print("\nРезультаты оптимизации:")
    print(f"Истинный оптимум: {target_point}")
    print(f"Найденное решение: {result}")
    print(f"Значение функции: {final_value:.6f}")
    print(f"Число вызовов модели: {calls_count}")
    print(f"Отклонение от оптимума: {np.linalg.norm(np.array(result) - target_point):.6f}")
    print(f"Отклонение значения функции: {(final_value - model.func(target_point)):.10f}")


if __name__ == "__main__":
    test_simulated_annealing()