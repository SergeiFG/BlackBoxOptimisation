import numpy as np
from BlackBoxOptimizer import SimulatedAnnealingOptimizer
from BlackBoxOptimizer import Optimizer

def quadratic_function(x):
    return np.array([np.sum(x**2)]) 

dim = 5  
iterations = 500
bounds = [(-10, 10) for _ in range(dim)]  

optimizer = Optimizer(
    optCls=SimulatedAnnealingOptimizer,
    to_model_vec_size=dim,
    from_model_vec_size=1,
    iter_limit=iterations,
    initial_temperature=100.0,
    cooling_rate=0.95
)

for i in range(dim):
    optimizer.setVecItemLimit(index=i, vec_dir="to_model", min=bounds[i][0], max=bounds[i][1])

optimizer.modelOptimize(quadratic_function)

best_params, best_value = optimizer.getResult()
history = optimizer.getHistoricalData("obj_val")

print(f"Лучшее решение: {best_params}")
print(f"Лучшее значение функции: {best_value}")
