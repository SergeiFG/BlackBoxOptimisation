from BlackBoxOptimizer import Optimizer
from BlackBoxOptimizer.BoxOptimizer.Genetic_Algo import Genetic_Algo
import numpy as np
from Models import SquareSumModel

class Constrained_Gen_Algo_test:
    def __init__(self):
        # Целевая точка (первые 3 параметра для SquareSumModel)
        self.target = np.array([135.123, 505.23121, -206.876, 0, 0])
        self.dimension = 5  # 3 непрерывных + 2 дискретных
        
        # Инициализация модели (работает только с первыми 3 параметрами)
        self.model = SquareSumModel(-self.target[:3])
        
        # Инициализация оптимизатора
        self.optimizer = Optimizer(
            optCls=Genetic_Algo,
            seed=42,
            to_model_vec_size=self.dimension,
            from_model_vec_size=1,  # SquareSumModel возвращает только ошибку
            iter_limit=50,
            external_model=self._adapted_model_evaluate,  # Используем адаптер
            # Параметры для EvolutionaryOpt
            population_size = 100,
            init_mutation = 0.5,
            min_mutation = 0.05,
            elite_size = 5,
            discrete_indices = [3, 4]
        )
        
        # Установка ограничений
        for i in range(3):  # Первые 3 параметра - непрерывные
            self.optimizer.setVecItemLimit(i, "to_model", min=-250, max=500)
         #   self.optimizer.setVecItemLimit(i, "to_model", min=-np.inf, max=np.inf)
        
        for i in range(3, 5):  # Последние 2 параметра - дискретные
            self.optimizer.setVecItemType(i, "bool", "to_model")
            self.optimizer.setVecItemLimit(i, "to_model", min=0, max=1)

    def _adapted_model_evaluate(self, x):
        """Адаптер для модели, передает только первые 3 параметра, потом добавляет по 100 к целевой
        функции за каждый неверный дискретный параметр"""
        discrete_error = 100 * np.sum(x[3:] != self.target[3:])
        return self.model.evaluate(x[:3]) + np.array(discrete_error)


    def run(self):
        print("=== Тест генетического алгоритма с ограничениями и дискретными параметрами ===")
        print(f"Целевые непрерывные значения: {self.target[:3]}")
        print(f"Целевые дискретные значения: {self.target[3:]}")
        print(f"Индексы дискретных параметров: [3, 4] (должны быть 0 или 1)")
            
        try:
            self.optimizer.modelOptimize()
            ep_optimizer = self.optimizer.getOptimizer()
            best_solution = ep_optimizer._to_opt_model_data.vecs[:, 0]
            
            final_error = self._adapted_model_evaluate(best_solution)
            
            print("\n=== Результаты ===")
            print(f"Лучшее решение: {np.round(best_solution, 4)}")
            print(f"Финальная ошибка: {final_error[0]}")
            print(f"Непрерывная часть: {best_solution[:3]}")
            print(f"Дискретная часть: {best_solution[3:]}")
            
            print("\nПроверка ограничений:")
            print(f"Непрерывные в пределах: {np.all(best_solution[:3] >= -250) & np.all(best_solution[:3] <= 500)}")
            print(f"Дискретные бинарные: {best_solution[3] in [0, 1] and best_solution[4] in [0, 1]}")

        except Exception as e:
            print(f"Ошибка оптимизации: {str(e)}")
            raise

if __name__ == "__main__":
    test = Constrained_Gen_Algo_test()
    test.run()