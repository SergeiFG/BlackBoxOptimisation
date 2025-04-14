from BlackBoxOptimizer import EvolutionaryOpt, Optimizer
from Example import ModelMinSquareSum
import numpy as np

class EvolutionaryOptTest:
    def __init__(self):
        # Целевая точка и параметры
        self.target = np.array([0.999, 0.990, -0.999])
        self.dimension = len(self.target)
        
        # Инициализация оптимизатора
        self.optimizer = Optimizer(
            optCls=EvolutionaryOpt,
            seed=42,
            to_model_vec_size=self.dimension,
            from_model_vec_size=1,
            iter_limit=100,
            dimension=self.dimension,
            population_size=30,
            offspring_per_parent=2,
            mutation_prob=0.3,
            sigma_init=0.2,
            t_max=100
        )
        
        # Установка ограничений для всех параметров
        for i in range(self.dimension):
            self.optimizer.setVecItemLimit(i, "to_model", min=-1.0, max=1.0)
        
        # Создаем модель
        self.model = ModelMinSquareSum(self.target)

    def run(self):
        print("=== Starting Evolutionary Optimization Test ===")
        print(f"Target: {self.target}")
        
        # Обертка для функции, возвращающая только ошибку
        def evaluate(x):
            return self.model.evaluate(x)[0]  # Берем только ошибку
            
        try:
            # Запуск оптимизации
            self.optimizer.modelOptimize(func=evaluate)
            
            # Получаем оптимизатор и его данные
            ep_optimizer = self.optimizer.getOptimizer()
            result = ep_optimizer._to_opt_model_data.vecs[:, 0]  # Получаем первый вектор
            
            if len(result) == self.dimension:
                error = np.sum((result - self.target)**2)
                print("\n=== Optimization Results ===")
                print(f"Function calls: {self.model.get_call_count()}")
                print(f"Best solution: {np.round(result, 4)}")
                print(f"Target point: {self.target}")
                print(f"Final error: {error:.6f}")
            else:
                print("Error: Optimization did not produce valid result")
                
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            raise

if __name__ == "__main__":
    test = EvolutionaryOptTest()
    test.run()