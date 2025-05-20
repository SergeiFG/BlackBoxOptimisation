import math
import numpy as np
from typing import Callable, Tuple, Optional, List
from ..BaseOptimizer import BaseOptimizer, _boolItem, _floatItem

class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    Улучшенная реализация алгоритма имитации отжига с поддержкой ограничений на выходные значения
    """

    def __init__(self, 
                 to_model_vec_size: int,
                 from_model_vec_size: int,
                 iter_limit: int,
                 seed: int = None,
                 initial_temp: float = 100.0,
                 min_temp: float = 1e-8,
                 cooling_rate: float = 0.99,
                 step_size: float = 1.0,
                 n_restarts: int = 3,
                 penalty_coef: float = 1e6,
                 **kwargs) -> None:
        
        super().__init__(
            to_model_vec_size=to_model_vec_size,
            from_model_vec_size=from_model_vec_size,
            iter_limit=iter_limit,
            seed=seed
        )
        
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.initial_step_size = step_size
        self.step_size = step_size
        self.n_restarts = n_restarts
        self.penalty_coef = penalty_coef
        
        self.best_solution = None
        self.best_energy = float('inf')
        self.best_output_values = None
        self.current_energy = float('inf')
        self.restart_counter = 0
        self.iteration_counter = 0
        
        self.acceptance_rate = 0.5
        self.target_acceptance = 0.4
        
        self._vec_candidates_size = 5

    def _get_from_model_limits(self):
        """Получить ограничения на все выходные значения из _from_model_data"""
        min_bounds = []
        max_bounds = []
        for prop in self._from_model_data._values_properties_list:
            min_bounds.append(prop.min if prop.min is not None else -np.inf)
            max_bounds.append(prop.max if prop.max is not None else np.inf)
        return min_bounds, max_bounds

    def _correct_discrete_values(self, solution: np.array) -> np.array:
        """Корректирует дискретные значения в решении"""
        corrected = solution.copy()
        for i in range(len(corrected)):
            props = self._to_opt_model_data._values_properties_list[i]
            if isinstance(props, _boolItem):
                corrected[i] = 1.0 if corrected[i] >= 0.5 else 0.0
            elif isinstance(props, _floatItem):
                corrected[i] = np.clip(corrected[i], props.min, props.max)
        return corrected

    def _check_output_constraints(self, output_values: np.ndarray) -> bool:
        """Проверка соблюдения ограничений на все выходные значения"""
        min_bounds, max_bounds = self._get_from_model_limits()
        for i in range(min(len(output_values), len(min_bounds))):
            if output_values[i] < min_bounds[i] or output_values[i] > max_bounds[i]:
                return False
        return True

    def _apply_penalty(self, energy: float, output_values: np.ndarray) -> float:
        """Применение штрафа к целевой функции при нарушении ограничений"""
        min_bounds, max_bounds = self._get_from_model_limits()
        penalty = 0.0
        for i in range(min(len(output_values), len(min_bounds))):
            if output_values[i] < min_bounds[i]:
                penalty += (min_bounds[i] - output_values[i])**2
            if output_values[i] > max_bounds[i]:
                penalty += (output_values[i] - max_bounds[i])**2
        return energy + self.penalty_coef * penalty

    def modelOptimize(self, func: Callable[[np.array], np.array]) -> None:
        """Основной метод оптимизации с поддержкой ограничений"""
        for restart in range(self.n_restarts):
            self._single_run_optimization(func)
            
            if restart < self.n_restarts - 1:
                self._prepare_restart()
                
    def _single_run_optimization(self, func: Callable[[np.array], np.array]) -> None:
        """Одиночный прогон алгоритма с учетом ограничений"""
        current_solutions = [self._correct_discrete_values(vec.copy()) 
                           for vec in self._to_opt_model_data.iterVectors()]
        
        current_outputs = []
        for sol in current_solutions:
            output = func(sol)
            print(f"Кандидат: {sol}, Значение функции: {output[0]}")
            current_outputs.append(output)
        current_energies = []
        
        for output in current_outputs:
            energy = output[0]
            if not self._check_output_constraints(output):
                energy = self._apply_penalty(energy, output)
            current_energies.append(energy)
        
        best_idx = np.argmin(current_energies)
        self.best_solution = current_solutions[best_idx].copy()
        self.best_energy = current_energies[best_idx]
        self.best_output_values = current_outputs[best_idx].copy()
        
        while self.iteration_counter < self._iteration_limitation and self.current_temp > self.min_temp:
            accepted = 0
            total = 0
            
            for i in range(len(current_solutions)):
                new_solution = self._generate_neighbor(current_solutions[i])
                new_output = func(new_solution)
                print(f"Кандидат: {new_solution}, Значение функции: {new_output[0]}")
                new_energy = new_output[0]
                total += 1
                
                if not self._check_output_constraints(new_output):
                    new_energy = self._apply_penalty(new_energy, new_output)
                
                if self._accept_solution(current_energies[i], new_energy):
                    current_solutions[i] = new_solution.copy()
                    current_energies[i] = new_energy
                    current_outputs[i] = new_output.copy()
                    accepted += 1
                    
                    if new_energy < self.best_energy:
                        self.best_solution = new_solution.copy()
                        self.best_energy = new_energy
                        self.best_output_values = new_output.copy()
            
            self._adapt_parameters(accepted / max(total, 1))
            self.iteration_counter += 1
            
            if self.iteration_counter % 50 == 0:
                self._synchronize_chains(current_solutions, current_energies, current_outputs)

    def _generate_neighbor(self, solution: np.array) -> np.array:
        """Генерация соседнего решения"""
        neighbor = solution.copy()
        for i in range(len(neighbor)):
            item_props = self._to_opt_model_data._values_properties_list[i]
            
            if isinstance(item_props, _boolItem):
                if np.random.random() < 0.1 * (1 - self.current_temp/self.initial_temp):
                    neighbor[i] = 1.0 - neighbor[i]
            elif isinstance(item_props, _floatItem):
                perturbation = np.random.normal(scale=self.step_size)
                neighbor[i] += perturbation
                if neighbor[i] < item_props.min:
                    neighbor[i] = 2 * item_props.min - neighbor[i]
                elif neighbor[i] > item_props.max:
                    neighbor[i] = 2 * item_props.max - neighbor[i]
        
        return self._correct_discrete_values(neighbor)
    
    def _accept_solution(self, current_energy: float, new_energy: float) -> bool:
        """Критерий принятия решения с защитой от переполнения"""
        if new_energy < current_energy:
            return True
        
        delta = new_energy - current_energy
        if delta == 0:
            return np.random.random() < 0.5
        
        try:
            probability = math.exp(-delta / max(self.current_temp, 1e-100))
            return np.random.random() < probability
        except OverflowError:
            return False

    def _adapt_parameters(self, current_acceptance: float):
        """Адаптация температуры и размера шага"""
        self.current_temp = self.initial_temp * math.exp(
            -10 * self.iteration_counter / self._iteration_limitation)
        
        if current_acceptance > self.target_acceptance:
            self.step_size *= 1.1
        else:
            self.step_size *= 0.9
            
        self.step_size = np.clip(self.step_size, 1e-4, 10 * self.initial_step_size)

    def _synchronize_chains(self, solutions, energies, outputs):
        """Обмен информацией между цепочками с учетом ограничений"""
        valid_indices = [i for i, out in enumerate(outputs) 
                        if self._check_output_constraints(out)]
        
        if valid_indices:
            best_valid_idx = min(valid_indices, key=lambda i: energies[i])
        else:
            best_valid_idx = np.argmin(energies)
        
        for i in range(len(solutions)):
            if i != best_valid_idx and np.random.random() < 0.1:
                for j in range(len(solutions[i])):
                    if np.random.random() < 0.3:
                        props = self._to_opt_model_data._values_properties_list[j]
                        if isinstance(props, _boolItem):
                            solutions[i][j] = float(bool(solutions[best_valid_idx][j]))
                        else:
                            solutions[i][j] = solutions[best_valid_idx][j]
                energies[i] = float('inf')

    def _prepare_restart(self):
        """Подготовка к рестарту алгоритма"""
        self.current_temp = self.initial_temp * 0.5  
        self.step_size = self.initial_step_size * 0.5
        self.iteration_counter = 0
        
        for i, vec in enumerate(self._to_opt_model_data.iterVectors()):
            if i == 0:
                vec[:] = self.best_solution
            else:
                vec[:] = self._generate_neighbor(self.best_solution)

    def getResult(self) -> np.array:
        return self.best_solution