from BlackBoxOptimizer import EvolutionaryOpt, Optimizer
import numpy as np
from Models import SquareSumModel

class ConstrainedEvolutionaryTest:
    def __init__(self):
        # Целевая точка и параметры
        self.target = np.array([2, 5, 10, 1, 0])  # 3 непрерывных + 2 дискретных
        self.dimension = len(self.target)
        self.model = SquareSumModel(-self.target[:3])  # Только первые 3 параметра
        
        # Инициализация оптимизатора
        self.optimizer = Optimizer(
            optCls=EvolutionaryOpt,
            seed=42,
            to_model_vec_size=self.dimension,
            from_model_vec_size=3,
            iter_limit=100,
            external_model=self._adapted_model_evaluate,
            dimension=self.dimension,
            population_size=30,
            offspring_per_parent=2,
            mutation_prob=0.3,
            sigma_init=0.2,
            t_max=100,
            discrete_indices=[3, 4]  # Индексы дискретных параметров
        )
        
        # Установка ограничений
        for i in range(3):  # Первые 3 параметра - непрерывные
            self.optimizer.setVecItemLimit(i, "to_model", min=-5, max=21)
        
        for i in range(3, 5):  # Последние 2 параметра - дискретные
            self.optimizer.setVecItemType(i, "bool", "to_model")
            self.optimizer.setVecItemLimit(i, "to_model", min=0, max=1)
        
        self.optimizer.setVecItemLimit(0, "from_model", min=0, max=np.inf)
        self.optimizer.setVecItemLimit(1, "from_model", min=0, max=2)
        self.optimizer.setVecItemLimit(2, "from_model", min=0, max=4)

    def _adapted_model_evaluate(self, x):
        """Адаптер для модели, передает только первые 3 параметра"""
        return self.model.evaluate(x[:3])

    def evaluate_full(self, x):
        """Полная оценка решения с учетом всех параметров"""
        model_error = self.model.evaluate(x[:3])[0]
        discrete_error = np.sum((x[3:] - self.target[3:])**2)
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
        print(f"Target point: {self.target}")
        print(f"Discrete parameters indices: [3, 4] (must be 0 or 1)")
            
        try:
            # Запуск оптимизации
            self.optimizer.modelOptimize()
            
            # Получаем оптимизатор и историю
            ep_optimizer = self.optimizer.getOptimizer()
            history = ep_optimizer.get_optimization_history()
            
            # Анализ истории
            print("\nАнализ истории оптимизации:")
            print(f"Всего поколений: {len(history)}")
            print(f"Начальное лучшее значение: {history[0]['best_fitness']:.4f}")
            print(f"Финальное лучшее значение: {history[-1]['best_fitness']:.4f}")
            print(f"Улучшение: {((history[0]['best_fitness'] - history[-1]['best_fitness'])/history[0]['best_fitness'])*100:.2f}%")
            
            # Визуализация истории
            ep_optimizer.plot_optimization_history()
            
            # Получаем результаты
            best_solution = ep_optimizer._to_opt_model_data.vecs[:, 0]
            constrained_solution = self.calculate_constrained_solution(best_solution)
            final_error = self.evaluate_full(constrained_solution)
            
            # Вывод результатов
            print("\n=== Results ===")
            print(f"Best solution: {np.round(best_solution, 4)}")
            print(f"Final error (constrained): {final_error:.6f}")
            print(f"Continuous part: {constrained_solution[:3]}")
            print(f"Discrete part: {constrained_solution[3:]}")
            
            # Проверка ограничений
            print("\nConstraints verification:")
            print(f"Continuous in bounds: {np.all(constrained_solution[:3] >= -5) & np.all(constrained_solution[:3] <= 21)}")
            print(f"Discrete are binary: {constrained_solution[3] in [0, 1] and constrained_solution[4] in [0, 1]}")

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            raise

if __name__ == "__main__":
    test = ConstrainedEvolutionaryTest()
    test.run()