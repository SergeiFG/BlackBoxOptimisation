from BlackBoxOptimizer import EvolutionaryOpt, Optimizer
import numpy as np

class ConstrainedEvolutionaryTest:
    def __init__(self):
        # Целевая точка и параметры
        self.target = np.array([2, 5, 10])
        self.dimension = len(self.target)
        
        # Инициализация оптимизатора
        self.optimizer = Optimizer(
            optCls=EvolutionaryOpt,
            seed=42,
            to_model_vec_size=self.dimension,
            from_model_vec_size=3,  # Ошибка + 2 дополнительных параметра
            iter_limit=100,
            dimension=self.dimension,
            population_size=30,
            offspring_per_parent=2,
            mutation_prob=0.3,
            sigma_init=0.2,
            t_max=100
        )
        
        # Установка ограничений для входных параметров
        for i in range(self.dimension):
            self.optimizer.setVecItemLimit(i, "to_model", min=-5, max=21)
        
        # Установка ограничений для выходных параметров:
        self.optimizer.setVecItemLimit(0, "from_model", min=0, max=np.inf)  # Ошибка
        self.optimizer.setVecItemLimit(1, "from_model", min=0, max=2)       # x[0]*2
        self.optimizer.setVecItemLimit(2, "from_model", min=0, max=4)       # x[1]*1

    def evaluate(self, x):
        """Целевая функция с учетом ограничений"""
        # Основная ошибка
        error = np.sum((x - self.target)**2)
        
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
        print("=== Constrained Evolutionary Optimization Test ===")
        print(f"Target point: {self.target}")
            
        try:
            # Запуск оптимизации
            self.optimizer.modelOptimize(func=self.evaluate)
            
            # Получаем результаты
            ep_optimizer = self.optimizer.getOptimizer()
            best_solution = ep_optimizer._to_opt_model_data.vecs[:, 0]
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
    test = ConstrainedEvolutionaryTest()
    test.run()