from BlackBoxOptimizer import Optimizer
from BlackBoxOptimizer.BoxOptimizer.Genetic_Algo import Genetic_Algo
from Example import ModelMinSquareSum
import numpy as np


class Genetic_Algo_Test:
    def __init__(self):
        # Целевая точка и параметры
        self.target = np.array([-1, 3, -4])
        self.dimension = len(self.target)
        
        # Инициализация оптимизатора
        self.optimizer = Optimizer(
            optCls=Genetic_Algo,
            seed=42,
            to_model_vec_size=self.dimension,
            from_model_vec_size=3,  # Теперь 3 выходных параметра: ошибка + 2 дополнительных
            iter_limit=100,
            dimension=self.dimension,
            population_size=300,
            generations=100,
            init_mutation=0.5,
            min_mutation=0.0001,
            elite_size=10,
            output_lower_bounds=np.array([0, 0, 0]),    # Нижние границы выходных параметров
            output_upper_bounds=np.array([np.inf, 2, 4]) # Верхние границы выходных параметров
        )
        
        # Установка ограничений для входных параметров
        for i in range(self.dimension):
            self.optimizer.setVecItemLimit(i, "to_model", min=-5, max=21)
        
        # Создаем модель
        self.model = ModelMinSquareSum(self.target)

    def evaluate(self, x):
        """Целевая функция с дополнительными выходными параметрами"""
        error = self.model.evaluate(x)[0]  # Основная ошибка
        
        # Дополнительные параметры с "отражением" от границ
        additional1 = np.clip(x[0] * 2, 0, 2)  # Ограничение [0, 2]
        additional2 = np.clip(x[1] * 1, 0, 4)  # Ограничение [0, 4]
        
        return np.array([error, additional1, additional2])

    def calculate_constrained_solution(self, x):
        """Вычисление 'реального' решения с учетом ограничений"""
        return np.array([
            np.clip(x[0], -5, 21),  # Ограничения входа
            np.clip(x[1], -5, 21),
            np.clip(x[2], -5, 21)
        ])

    def run(self):
        print("=== Constrained Genetic Algorithm Optimization Test ===")
        print(f"Target point: {self.target}")
            
        try:
            # Запуск оптимизации с новой функцией evaluate
            self.optimizer.modelOptimize(func=self.evaluate)
            
            # Получаем оптимизатор и его данные
            ga_optimizer = self.optimizer.getOptimizer()
            best_solution = ga_optimizer._to_opt_model_data.vecs[:, 0]
            constrained_solution = self.calculate_constrained_solution(best_solution)
            
            # Расчет итоговой ошибки с учетом ограничений
            final_error = np.sum((constrained_solution - self.target)**2)
            
            # Вывод результатов
            print("\n=== Results ===")
            print(f"Best solution: {np.round(best_solution, 4)}")
            print(f"Final error (constrained): {final_error:.6f}")
            
            # Детализация выходных параметров
            output = self.evaluate(constrained_solution)
            print("\nOutput parameters:")
            print(f"  x[0]*2 = {output[1]:.4f} (must be 0-2)")
            print(f"  x[1]*1 = {output[2]:.4f} (must be 0-4)")
            
            # Проверка ограничений
            print("\nConstraints verification:")
            print(f"Input constraints: {np.all(constrained_solution >= -5) & np.all(constrained_solution <= 21)}")
            print(f"Output constraints:")
            print(f"  x[0]*2 in [0,2]: {0 <= output[1] <= 2}")
            print(f"  x[1]*1 in [0,4]: {0 <= output[2] <= 4}")

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            raise


if __name__ == "__main__":
    test = Genetic_Algo_Test()
    test.run()