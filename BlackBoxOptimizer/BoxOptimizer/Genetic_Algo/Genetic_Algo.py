from typing import Callable, Optional, List
import numpy as np
from ..BaseOptimizer import BaseOptimizer
from .OptClass import GeneticAlgorithmOptimizer




class Genetic_Algo(BaseOptimizer):
    def __init__(self, 
                to_model_vec_size: int,
                from_model_vec_size: int,
                iter_limit: int = 50,
                seed: int = None,
                population_size: int = 500,
                init_mutation: float = 0.1,
                min_mutation: float = 0.001,
                elite_size: int = 10,
                discrete_indices = None,
                base_penalty: int = 1e6, 
                adaptive_penalty: bool = True,
                penalty_exponent: int = 2, 
                feasibility_phase_generations: float = 0.3,
                **kwargs) -> None:
        super().__init__(
            to_model_vec_size=to_model_vec_size,
            from_model_vec_size=from_model_vec_size,
            iter_limit=iter_limit,
            seed=seed
        )
        self.seed = seed
        self.dimension = to_model_vec_size
        self.population_size = population_size
        self.generations = iter_limit
        self.init_mutation = init_mutation
        self.min_mutation = min_mutation
        self.elite_size = elite_size
        self.discrete_indices = discrete_indices if discrete_indices is not None else []
        self.base_penalty = base_penalty
        self.adaptive_penalty = adaptive_penalty
        self.penalty_exponent = penalty_exponent
        self.feasibility_phase_generations = feasibility_phase_generations
        self.optimization_history = None

    def configure(self, **kwargs):
        super().configure(**kwargs)
        for key in ['dimension', 'output_lower_bounds', 'output_upper_bounds', 'discrete_indices']:
            if key in kwargs:
                setattr(self, key, kwargs[key])

    def _get_bounds(self, vec_dir: str):
        """Получаем ограничения для указанного направления"""
        vec_data = self._to_opt_model_data if vec_dir == "to_model" else self._from_model_data
        size = self._to_model_vec_size if vec_dir == "to_model" else self._from_model_vec_size
        
        lower = np.full(size, -np.inf)
        upper = np.full(size, np.inf)
        
        # Безопасный доступ к границам через properties_list
        for i in range(size):
            if i < len(vec_data._values_properties_list):
                lower[i] = vec_data._values_properties_list[i].min
                upper[i] = vec_data._values_properties_list[i].max
            
        return lower, upper
    

    def modelOptimize(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        input_lower, input_upper = self._get_bounds("to_model")
        output_lower, output_upper = self._get_bounds("from_model")

        # Создаем экземпляр оптимизатора
        ga = GeneticAlgorithmOptimizer(
            func=func,
            dimension=self._to_model_vec_size,
            population_size=self.population_size,
            generations=self.generations,
            init_mutation=self.init_mutation,
            min_mutation=self.min_mutation,
            elite_size=self.elite_size,
            lower_bounds=input_lower,
            upper_bounds=input_upper,
            output_lower_bounds=output_lower,
            output_upper_bounds=output_upper,
            discrete_indices=self.discrete_indices,
            base_penalty=self.base_penalty,
            adaptive_penalty=self.adaptive_penalty,
            penalty_exponent=self.penalty_exponent,
            feasibility_phase_generations=self.feasibility_phase_generations
        )
        
        # Запускаем оптимизацию
        best_x, best_f, best_outputs = ga.run()
        
        # Сохраняем историю
        self.optimization_history = []
        for gen in range(self.generations):
            history_entry = {
                'generation': gen,
                'best_MV': ga.best_individuals_history[gen],
                'best_CV': ga.output_values_history[gen],
                'valid_solutions': ga.num_valid_solutions_history[gen],
                'average_fitness': ga.avg_fitness_history[gen]
            }
            self.optimization_history.append(history_entry)
        
        # Записываем best_x в контейнер BaseOptimizer
        for to_vec in self._to_opt_model_data.iterVectors():
            to_vec[:] = best_x


    def get_optimization_history(self) -> List[dict]:
        """Возвращает историю оптимизации в виде списка словарей"""
        return self.optimization_history

    def get_history_as_dataframe(self):
        """Возвращает историю в виде DataFrame (требует pandas)"""
        import pandas as pd
        return pd.DataFrame(self.optimization_history)

    def plot_optimization_history(self):
        """Визуализирует историю оптимизации"""
        if not self.optimization_history:
            raise ValueError("История оптимизации пуста. Сначала запустите modelOptimize.")
            
        import matplotlib.pyplot as plt
        
        # Создаем фигуру с несколькими субграфиками
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # График 1: Целевая функция и допустимые решения
        generations = [e['generation'] for e in self.optimization_history]
        avg_fitness = [e['average_fitness'] for e in self.optimization_history]
        valid_solutions = [e['valid_solutions'] for e in self.optimization_history]
        
        ax1.plot(generations, avg_fitness, 'b-', label='Среднее значение CV0')
        ax1.set_ylabel('Целевая функция (CV0)')
        ax1.set_title('История оптимизации')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        ax1b = ax1.twinx()
        ax1b.plot(generations, valid_solutions, 'r-', label='Допустимые решения')
        ax1b.set_ylabel('Количество', color='r')
        ax1b.tick_params(axis='y', labelcolor='r')
        ax1b.set_ylim(0, self.population_size * 1.1)
        ax1b.legend(loc='upper right')
        
        # График 2: Выходные параметры (CV)
        if len(self.optimization_history) > 0 and 'best_CV' in self.optimization_history[0]:
            n_cv = len(self.optimization_history[0]['best_CV'])
            
            # Для каждого CV создаем график
            for i in range(n_cv):
                cv_values = [e['best_CV'][i] for e in self.optimization_history]
                ax2.plot(generations, cv_values, label=f'CV{i}')
            
            ax2.set_xlabel('Поколение')
            ax2.set_ylabel('Значение CV')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def getResult(self) -> np.ndarray:
        historical_data = self.getHistoricalData("vec_to_model")
        if historical_data and len(historical_data) > 0:
            return historical_data[-1]
        return np.array([])