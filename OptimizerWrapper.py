#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import numpy as np
from BlackBoxOptimizer import TestStepOpt, Optimizer

class CustomOptimizer:
    def __init__(self, objective_function, mv_id, bounds, step, max_iter):
        self.objective_function = objective_function
        self.mv_id = mv_id
        self.bounds = bounds
        self.step = step
        self.max_iter = max_iter
        self.current_value = 0.0
        self.iteration = 0
        
    def evaluate(self, x):
        # Подставляем значение в целевую функцию
        expr = self.objective_function.replace(f"[{self.mv_id}]", str(x[0]))
        
        try:
            value = eval(expr, {
                'Pow': pow, 'pow': pow,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'sqrt': np.sqrt, 'log10': np.log10
            })
            return np.array([value])
        except Exception as e:
            raise ValueError(f"Evaluation error: {str(e)}")
    
    def optimize(self):
        best_value = self.current_value
        best_score = self.evaluate([best_value])[0]
        
        for _ in range(self.max_iter):
            # Простейший градиентный спуск
            current_score = self.evaluate([self.current_value])[0]
            
            # Пробуем шаг вправо
            right_val = min(self.current_value + self.step, self.bounds[1])
            right_score = self.evaluate([right_val])[0]
            
            # Пробуем шаг влево
            left_val = max(self.current_value - self.step, self.bounds[0])
            left_score = self.evaluate([left_val])[0]
            
            # Выбираем наилучшее направление
            if right_score < left_score and right_score < current_score:
                self.current_value = right_val
                best_score = right_score
            elif left_score < current_score:
                self.current_value = left_val
                best_score = left_score
            
            self.iteration += 1
        
        return [self.current_value]

def run_optimization(params):
    try:
        # Получаем параметры
        mv_id = params['mv_ids'][0]
        bounds = params['bounds'][0]
        objective = params['objective_function']
        
        # Создаем и запускаем оптимизатор
        optimizer = CustomOptimizer(
            objective_function=objective,
            mv_id=mv_id,
            bounds=bounds,
            step=0.01,  # Фиксированный шаг
            max_iter=params['iter_limit']
        )
        
        # Получаем результат
        result = optimizer.optimize()
        
        output = {
            "optimized_values": result,
            "iterations_used": optimizer.iteration,
            "status": "success"
        }
        print("FINAL_RESULT:" + json.dumps(output))
    
    except Exception as e:
        print("ERROR:" + json.dumps({
            "error": str(e),
            "status": "failed"
        }))
        exit(1)

if __name__ == "__main__":
    try:
        run_optimization(json.loads(input()))
    except Exception as e:
        print("CRITICAL_ERROR:" + str(e))
        exit(1)