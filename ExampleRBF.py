from BlackBoxOptimizer import EvolutionaryOpt, Optimizer
import numpy as np
from Models import SquareSumModel

class ConstrainedEvolutionaryTest:
    def __init__(self):
        # Целевая точка (первые 3 параметра для SquareSumModel)
        self.target_continuous = np.array([2, 5, 10])
        self.target_discrete = np.array([1, 0])  # Дополнительные дискретные параметры
        self.dimension = 5  # 3 непрерывных + 2 дискретных
        
        # Инициализация модели (работает только с первыми 3 параметрами)
        self.model = SquareSumModel(-self.target_continuous)
        
        # Инициализация оптимизатора
        self.optimizer = Optimizer(
            optCls=EvolutionaryOpt,
            seed=42,
            to_model_vec_size=self.dimension,
            from_model_vec_size=1,  # SquareSumModel возвращает только ошибку
            iter_limit=100,
            external_model=self._adapted_model_evaluate,  # Используем адаптер
            # Параметры для EvolutionaryOpt
            opt_params={
                'dimension': self.dimension,
                'population_size': 30,
                'offspring_per_parent': 2,
                'mutation_prob': 0.3,
                'sigma_init': 0.2,
                't_max': 100,
                'discrete_indices': [3, 4]  # Индексы дискретных параметров
            }
        )
        
        # Установка ограничений
        for i in range(3):  # Первые 3 параметра - непрерывные
            self.optimizer.setVecItemLimit(i, "to_model", min=-5, max=21)
        
        for i in range(3, 5):  # Последние 2 параметра - дискретные
            self.optimizer.setVecItemType(i, "bool", "to_model")
            self.optimizer.setVecItemLimit(i, "to_model", min=0, max=1)

    def _adapted_model_evaluate(self, x):
        """Адаптер для модели, передает только первые 3 параметра"""
        return self.model.evaluate(x[:3])

    def evaluate_full(self, x):
        """Полная оценка решения с учетом всех параметров"""
        model_error = self.model.evaluate(x[:3])[0]  # Ошибка от SquareSumModel
        discrete_error = np.sum((x[3:] - self.target_discrete)**2)  # Ошибка дискретных параметров
        return model_error + discrete_error

    def calculate_constrained_solution(self, x):
        """Вычисление 'реального' решения с учетом ограничений"""
        constrained = np.array([
            np.clip(x[0], -5, 21),
            np.clip(x[1], -5, 21),
            np.clip(x[2], -5, 21)
        ])
        discrete = np.array([1 if x[3] >= 0.5 else 0, 1 if x[4] >= 0.5 else 0])
        return np.concatenate([constrained, discrete])

    def run(self):
        print("=== Constrained Evolutionary Optimization Test with Discrete Parameters ===")
        print(f"Target continuous: {self.target_continuous}")
        print(f"Target discrete: {self.target_discrete}")
        print(f"Discrete parameters indices: [3, 4] (must be 0 or 1)")
            
        try:
            self.optimizer.modelOptimize()
            ep_optimizer = self.optimizer.getOptimizer()
            best_solution = ep_optimizer._to_opt_model_data.vecs[:, 0]
            constrained_solution = self.calculate_constrained_solution(best_solution)
            
            final_error = self.evaluate_full(constrained_solution)
            
            print("\n=== Results ===")
            print(f"Best solution: {np.round(best_solution, 4)}")
            print(f"Final error (constrained): {final_error:.6f}")
            print(f"Continuous part: {constrained_solution[:3]}")
            print(f"Discrete part: {constrained_solution[3:]}")
            
            print("\nConstraints verification:")
            print(f"Continuous in bounds: {np.all(constrained_solution[:3] >= -5) & np.all(constrained_solution[:3] <= 21)}")
            print(f"Discrete are binary: {constrained_solution[3] in [0, 1] and constrained_solution[4] in [0, 1]}")

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            raise

if __name__ == "__main__":
    test = ConstrainedEvolutionaryTest()
    test.run()