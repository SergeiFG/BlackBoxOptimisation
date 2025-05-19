from BlackBoxOptimizer import Optimizer, OptimisationTypes
from Models import SquareSumModel
import numpy as np
from BlackBoxOptimizer import SimulatedAnnealingOptimizer

def test_simulated_annealing():
    target_point = np.array([101, 50, 72])
    model = SquareSumModel(-target_point)
    
    opt = Optimizer(
        optCls=SimulatedAnnealingOptimizer,
        seed=1424,
        to_model_vec_size=3,
        from_model_vec_size=1,
        iter_limit=100000,  
        external_model=model.evaluate,
        optimisation_type=OptimisationTypes.minimize,
        initial_temp=100.0, 
        min_temp=1e-5,       
        cooling_rate=0.999, 
        step_size=1,      
    )
    
    opt.setVecItemLimit(0, min=0, max=1000)
    opt.setVecItemLimit(1, min=0, max=1000)
    opt.setVecItemLimit(2, min=0, max=1000)
    

    initial_guess = np.array([0, 0, 0])  
    opt.setPreSetCadidateVec(0, initial_guess)
    

    opt.modelOptimize()
    
    # Получаем результаты
    result = opt.getResult()
    final_value = model.evaluate(result)[0]
    calls_count = opt.get_usage_count()
    
    print("\nРезультаты оптимизации:")
    print(f"Истинный оптимум: {target_point}")
    print(f"Найденное решение: {result}")
    print(f"Значение функции: {final_value:.6f}")
    print(f"Число вызовов модели: {calls_count}")
    print(f"Отклонение от оптимума: {np.linalg.norm(result - target_point):.6f}")
    print(f"Отклонение значения функции: {(final_value - model.func(target_point)):.10f}")

if __name__ == "__main__":
    test_simulated_annealing()
from BlackBoxOptimizer import Optimizer, OptimisationTypes
from Models import SquareSumModel
import numpy as np
from BlackBoxOptimizer import SimulatedAnnealingOptimizer

def test_simulated_annealing():
    target_point = np.array([1, 2, 3])  
    model = SquareSumModel(-target_point)
    
    print(f"Истинный оптимум: {target_point}")
    print(f"Минимальное значение функции: {model.func(target_point):.4f}\n")
    
    opt = Optimizer(
        optCls=SimulatedAnnealingOptimizer,
        seed=1424,
        to_model_vec_size=3,
        from_model_vec_size=1,
        iter_limit=15000, 
        external_model=model.evaluate,
        optimisation_type=OptimisationTypes.minimize,
        initial_temp=10.0,  #Ставить примерно от 500 при больших значениях
        min_temp=1e-12,       
        cooling_rate=0.99,   
        step_size=5.0        #Ставить меньше при мальеньких значениях
    )
    
    opt.setVecItemLimit(0, min=-10, max=2000)
    opt.setVecItemLimit(1, min=-10, max=5000)
    opt.setVecItemLimit(2, min=-10, max=7000)
    
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