from BlackBoxOptimizer import Optimizer
from BlackBoxOptimizer.BoxOptimizer.Genetic_Algo import Genetic_Algo
import numpy as np
from Models import SquareSumModel

class GeneticAlgoTest:
    def __init__(self):
        # Целевая точка (первые 3 параметра для SquareSumModel)
        self.target_continuous = np.array([-7, 20, 10, 7, 26])
        self.target_discrete = np.array([0, 0])  # Дополнительные дискретные параметры
        self.dimension = 7  # 3 непрерывных + 2 дискретных
        
        # Инициализация модели (работает только с первыми 3 параметрами)
        self.model = SquareSumModel(-self.target_continuous)
        
        # Инициализация оптимизатора
        self.optimizer = Optimizer(
            optCls=Genetic_Algo,
            seed=42,
            to_model_vec_size=self.dimension,
            from_model_vec_size=1,  # SquareSumModel возвращает только ошибку
            iter_limit=100,
            external_model=self._adapted_model_evaluate,  # Используем адаптер
            # Параметры для GeneticAlgoritm
            opt_params={
                'dimension': self.dimension,
                'population_size': 200,
                'generations': 50,
                'init_mutation': 0.6,
                'min_mutation': 0.01,
                'elite_size': 5,
                'output_lower_bounds': [0, 0, 0, 0, 0],
                'output_upper_bounds': [np.inf, 2, 4, np.inf, np.inf],
                'discrete_indices': [5, 6]  # Булевые параметры
            }
        )
        
        # Установка ограничений
        for i in range(5):  # Первые 5 параметра - непрерывные
            self.optimizer.setVecItemLimit(i, "to_model", min=-5, max=21)
        
        for i in range(5, 7):  # Последние 2 параметра - дискретные
            self.optimizer.setVecItemType(i, "bool", "to_model")
            self.optimizer.setVecItemLimit(i, "to_model", min=0, max=1)

    def _adapted_model_evaluate(self, x):
        """Адаптер для модели, передает только первые 3 параметра"""
        return self.model.evaluate(x[:5])

    def evaluate_full(self, x):
        """Полная оценка решения с учетом всех параметров"""
        model_error = self.model.evaluate(x[:5])[0]  # Ошибка от SquareSumModel
        discrete_error = np.sum((x[5:] - self.target_discrete)**2)  # Ошибка дискретных параметров
        return model_error + discrete_error

    def calculate_constrained_solution(self, x):
        """Вычисление 'реального' решения с учетом ограничений"""
        constrained = np.array([
            np.clip(x[0], -5, 21),
            np.clip(x[1], -5, 21),
            np.clip(x[2], -5, 21),
            np.clip(x[3], -5, 21),
            np.clip(x[4], -5, 21)
        ])
        discrete = np.array([1 if x[5] >= 0.5 else 0, 1 if x[6] >= 0.5 else 0])
        return np.concatenate([constrained, discrete])

    def run(self):
        print("=== Тест генетического алгоритма с дискретными и непрерывными параметрами ===")
        print(f"Целевые значения непрерывных параметров: {self.target_continuous}")
        print(f"Целевые значения дискретных параметров: {self.target_discrete}")
            
        try:
            self.optimizer.modelOptimize()
            ep_optimizer = self.optimizer.getOptimizer()
            best_solution = ep_optimizer._to_opt_model_data.vecs[:, 0]
            constrained_solution = self.calculate_constrained_solution(best_solution)
            
            final_error = self.evaluate_full(constrained_solution)
            
            print("\n=== Результат ===")
            print(f"Лучший результат: {np.round(best_solution, 4)}")
            print(f"Суммарная ошибка лучшего результата: {final_error:.6f}")
            print(f"Непрерывные параметры лучшего результата: {constrained_solution[:5]}")
            print(f"Дискретные параметры лучшего результата: {constrained_solution[5:]}")
            
            print("\nПроверка соблюдения границ:")
            print(f"Непрерывные параметры в границах: {np.all(constrained_solution[:5] >= -5) & np.all(constrained_solution[:5] <= 21)}")
            print(f"Дискретные параметры принимают значения 0 или 1: {constrained_solution[5] in [0, 1] and constrained_solution[6] in [0, 1]}")

        except Exception as e:
            print(f"Ошибка оптимизации: {str(e)}")
            raise

if __name__ == "__main__":
    test = GeneticAlgoTest()
    test.run()