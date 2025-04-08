import numpy as np
import math
from typing import Callable
import sys
sys.path.append(r"C:\Users\PC\Desktop\ITK_RTO")
from BlackBoxOptimizer.BoxOptimizer.BaseOptimizer import BaseOptimizer, OptimizedVectorData

class SimulatedAnnealingOptimizer(BaseOptimizer):
    def __init__(self,
                 to_model_vec_size: int,
                 from_model_vec_size: int,
                 iter_limit: int,
                 main_value_index: int = 0,
                 initial_temperature: float = 100.0,
                 cooling_rate: float = 0.95,
                 min_temperature: float = 1e-3) -> None:
        
        super().__init__(
            to_model_vec_size=to_model_vec_size,
            from_model_vec_size=from_model_vec_size,
            iter_limit=iter_limit,
            main_value_index=main_value_index
        )
        
        self._initial_temperature = initial_temperature
        self._temperature = initial_temperature
        self._cooling_rate = cooling_rate
        self._min_temperature = min_temperature
        
        self._best_solution = None
        self._best_value = np.inf
        self._current_solution = None
        self._current_value = np.inf

    def _generate_neighbor(self) -> np.array:
        neighbor = self._current_solution.copy()
        perturbation = np.random.normal(0, 1, size=self._to_model_vec_size) * self._temperature
        neighbor += perturbation
        
        # Проверка ограничений
        for i in range(self._to_model_vec_size):
            min_val = self._to_opt_model_data._vec[i, OptimizedVectorData.min_index]
            max_val = self._to_opt_model_data._vec[i, OptimizedVectorData.max_index]
            neighbor[i] = np.clip(neighbor[i], min_val, max_val)
                
        return neighbor

    def _acceptance_probability(self, old_value: float, new_value: float) -> float:
        if new_value < old_value:
            return 1.0
        return math.exp((old_value - new_value) / self._temperature)

    def _calc_objective_function_value(self) -> None:
        pass

    def _main_calc_func(self) -> None:
        if self._temperature < self._min_temperature:
            return
            
        self._current_value = self._from_model_data._vec[0, OptimizedVectorData.values_index_start]
        
        if self._current_value < self._best_value:
            self._best_value = self._current_value
            self._best_solution = self._current_solution.copy()
            
        neighbor = self._generate_neighbor()
        
        temp_vec = self._to_opt_model_data._vec[:, OptimizedVectorData.values_index_start].copy()
        self._to_opt_model_data._vec[:, OptimizedVectorData.values_index_start] = neighbor

        output = self._objective_function(neighbor)
        self._from_model_data._vec[0, OptimizedVectorData.values_index_start] = output
        neighbor_value = output
        
        if self._acceptance_probability(self._current_value, neighbor_value) > np.random.random():
            self._current_solution = neighbor
            self._current_value = neighbor_value
        else:
            self._to_opt_model_data._vec[:, OptimizedVectorData.values_index_start] = temp_vec
            
        self._temperature *= self._cooling_rate

    def modelOptimize(self, func: Callable[[np.array], np.array]) -> None:
        self._objective_function = func
        
        self._current_solution = self._to_opt_model_data._vec[:, OptimizedVectorData.values_index_start].copy()
        
        output = func(self._current_solution)
        self._from_model_data._vec[0, OptimizedVectorData.values_index_start] = output
        self._current_value = output
        self._best_solution = self._current_solution.copy()
        self._best_value = self._current_value
        
        for _ in range(self._iteration_limitation):
            if self._temperature < self._min_temperature:
                break
                
            self._main_calc_func()

    def getResult(self) -> tuple[np.array, float]:
        return self._best_solution, self._best_value
    
def cracking_efficiency(params):
    """
    params: [temperature (C), pressure (atm), catalyst_activity (0-1)]
    Возвращает выход бензина в % (чем больше - тем лучше)
    """
    temp, pressure, catalyst = params
    
    # Имитация реального процесса:
    # - Оптимальная температура 450-550°C
    # - Оптимальное давление 10-20 атм
    # - Катализатор улучшает выход
    
    base_yield = 40  # Базовый выход
    
    # Температурная зависимость (колокол вокруг 500°C)
    temp_effect = 30 * np.exp(-0.0003*(temp-500)**2)
    
    # Давление (оптимум 15 атм)
    pressure_effect = 10 * (1 - abs(pressure-15)/15)
    
    # Влияние катализатора
    catalyst_effect = 20 * catalyst
    
    total_yield = base_yield + temp_effect + pressure_effect + catalyst_effect
    

    return -total_yield  # Минимизируем отрицательный выход

optimizer = SimulatedAnnealingOptimizer(
    to_model_vec_size=3,  # [temp, pressure, catalyst]
    from_model_vec_size=1,
    iter_limit=500,
    initial_temperature=200,
    cooling_rate=0.9,
    min_temperature=0.1
)

data = optimizer._to_opt_model_data

# 1. Температура крекинга (°C)
data.setLimitation(0, min=300, max=600)  # Реальные границы процесса

# 2. Давление (атм)
data.setLimitation(1, min=5, max=30)     # Диапазон рабочего давления

# 3. Активность катализатора (0-1)
data.setLimitation(2, min=0.1, max=0.95) # Катализатор не может быть 100% активным


initial_temp = 450
initial_pressure = 12
initial_catalyst = 0.5

data._vec[:, data.values_index_start] = [
    initial_temp,
    initial_pressure,
    initial_catalyst
]

optimizer.modelOptimize(cracking_efficiency)
best_params, best_yield = optimizer.getResult()


optimal_temp, optimal_pressure, optimal_catalyst = best_params
real_yield = -best_yield  

print(f"Оптимальные параметры:")
print(f"Температура: {optimal_temp:.1f} °C")
print(f"Давление: {optimal_pressure:.1f} атм")
print(f"Активность катализатора: {optimal_catalyst:.2f}")
print(f"Прогнозируемый выход бензина: {real_yield:.1f}%")