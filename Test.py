from BlackBoxOptimizer import Optimizer, OptimisationTypes
from Models import SquareSumModel
import numpy as np
from BlackBoxOptimizer import SimulatedAnnealingOptimizer

def test_simulated_annealing():
    target_point = np.array([7, 5, 5])  
    model = SquareSumModel(-target_point)
    
    print(f"Истинный оптимум: {target_point}")
    print(f"Минимальное значение функции: {model.func(target_point):.4f}\n")
    
    opt = Optimizer(
        optCls=SimulatedAnnealingOptimizer,
        seed=42,
        to_model_vec_size=3,
        from_model_vec_size=1,
        iter_limit=2000, 
        external_model=model.evaluate,
        optimisation_type=OptimisationTypes.minimize,
        initial_temp=100.0,  
        min_temp=1e-20,       
        cooling_rate=0.95,   
        step_size=1.0        
    )
    
    opt.setVecItemLimit(0, min=-200, max=100)
    opt.setVecItemLimit(1, min=-200, max=100)
    opt.setVecItemLimit(2, min=-200, max=100)
    
    opt.modelOptimize()
    

    result = opt.getResult()
    final_value = model.evaluate(result)[0]
    calls_count = opt.get_usage_count()
    
    print("\nРезультаты оптимизации:")
    print(f"Найденное решение: {result}")
    print(f"Значение функции: {final_value:.6f}")
    print(f"Число вызовов модели: {calls_count}")
    print(f"Отклонения от значения функции: {(final_value - model.func(target_point)):.10f}")
    

if __name__ == "__main__":
    test_simulated_annealing()