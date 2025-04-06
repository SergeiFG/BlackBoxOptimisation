import sys
import os
import numpy as np
from typing import Tuple

# Настройка путей
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from BlackBoxOptimizer.BoxOptimizer.Optimizer import Optimizer
from BlackBoxOptimizer.BoxOptimizer.EvolutionaryOpt.EvolutionaryOpt import EvolutionaryOpt

class ModelMinSquareSum:
    def __init__(self, target: np.ndarray):
        self.target = target
        self._call_count = 0
        self._max_calls = 1000  # Лимит вызовов функции

    def evaluate(self, to_vec: np.ndarray, *args, **kwargs):
        """Вычисление целевой функции с ограничением количества вызовов"""
        self._call_count += 1
        
        if self._call_count > self._max_calls:
            raise StopIteration(f"Достигнут лимит в {self._max_calls} вызовов")
        
        error = np.sum((to_vec - self.target) ** 2)
        
        # Для демонстрации выводим каждый 50-й вызов
        if self._call_count % 50 == 0:
            print(f"Call #{self._call_count}: Input {np.round(to_vec, 4)} → Error {error:.4f}")
        
        return error

    def get_call_count(self):
        return self._call_count

if __name__ == "__main__":
    print("\n=== Starting RBF Optimization Test ===")
    
    # Параметры теста
    target_point = np.array([0, 0.5, -0.2])
    dimension = len(target_point)
    print(f"Target point: {target_point}")
    print(f"Dimension: {dimension}")

    # Создаем модель
    model = ModelMinSquareSum(target_point)
    
    # Конфигурация оптимизатора
    opt = Optimizer(
        optCls=EvolutionaryOpt,
        seed=42,
        to_model_vec_size=dimension,
        from_model_vec_size=1,
        iter_limit=200,
        dimension=dimension,
        population_size=20,
        offspring_per_parent=3,
        mutation_prob=0.2,
        sigma_init=0.15,
        t_max=200
    )

    print("\nOptimization parameters:")
    print(f"Population size: {opt._CurrentOptimizerObject.population_size}")
    print(f"Max iterations: {opt._CurrentOptimizerObject.t_max}")
    print(f"Mutation probability: {opt._CurrentOptimizerObject.mutation_prob}")

    try:
        print("\nStarting optimization...")
        opt.modelOptimize(func=model.evaluate)
        
        # Получаем результаты
        optimizer = opt.getOptimizer()
        best_solution = optimizer.getResult()
        history_points, history_values = optimizer.get_history()
        
        print("\n=== Optimization Results ===")
        print(f"Total function calls: {model.get_call_count()}")
        print(f"Best solution found: {np.round(best_solution, 4)}")
        print(f"Target point:       {target_point}")
        print(f"Difference:         {np.round(best_solution - target_point, 4)}")
        print(f"Final error:        {np.sum((best_solution - target_point)**2):.6f}")
        
        # Анализ истории
        print("\nOptimization history analysis:")
        print(f"Total points in history: {len(history_points)}")
        
        if len(history_points) > 0:
            print("\nBest 5 points in history:")
            best_indices = np.argsort(history_values)[:5]
            for idx in best_indices:
                print(f"Iter {idx}: {np.round(history_points[idx], 4)} " 
                      f"(error: {history_values[idx]:.6f})")
            
            print("\nConvergence progress:")
            print(f"Initial error: {history_values[0]:.6f}")
            print(f"Final error:   {history_values[-1]:.6f}")
            print(f"Improvement:   {100*(history_values[0]-history_values[-1])/history_values[0]:.1f}%")
            
    except StopIteration as e:
        print(f"\nOptimization stopped: {e}")
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        raise

    print("\n=== Test completed ===")