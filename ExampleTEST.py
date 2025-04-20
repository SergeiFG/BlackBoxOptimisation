#!/usr/bin/env python
# -*- coding: utf-8 -*-
from BlackBoxOptimizer import TestStepOpt
from BlackBoxOptimizer import Optimizer
import numpy as np
import json
import sys
from typing import Tuple

class BaseExternalModel:
    def __init__(self):
        self._call_count = 0

    def _increment_call_count(self):
        self._call_count += 1

    def evaluate(self, to_vec: np.ndarray, *args, **kwargs):
        raise NotImplementedError("Метод evaluate должен быть переопределен в подклассе")

    def _track_evaluate_calls(func):
        def wrapper(self, vec):
            self._increment_call_count()
            return func(self, vec)
        return wrapper

    def get_call_count(self):
        return self._call_count

class ModelMinSquareSum(BaseExternalModel):
    def __init__(self, target: np.ndarray):
        super().__init__()
        self.target = target

    @BaseExternalModel._track_evaluate_calls
    def evaluate(self, to_vec, *args, **kwargs):
        return np.array([np.sum((to_vec - self.target) ** 2), max(to_vec)])



def main():
    try:
        # Чтение входных данных с обработкой кодировки
        input_bytes = sys.stdin.buffer.read()
        input_str = input_bytes.decode('utf-8')
        params = json.loads(input_str)
        
        # Проверка параметров
        if not all(k in params for k in ['to_model_vec_size', 'initial_values']):
            raise ValueError("Missing required parameters")
            
        # Инициализация оптимизатора
        opt = Optimizer(
            optCls=TestStepOpt,
            seed=1546,
            to_model_vec_size=params['to_model_vec_size'],
            from_model_vec_size=params.get('from_model_vec_size', 2),
            iter_limit=params.get('iter_limit', 100)
        )
        opt.configure(step=0.01)
        
        # Запуск оптимизации
        target = np.array(params.get('target_point', [0.0]*params['to_model_vec_size']))
        model = ModelMinSquareSum(target)
        opt.modelOptimize(func=model.evaluate)
        
        # Формирование результата
        optimizer = opt.getOptimizer()
        result = {
            'optimized_values': optimizer.getResult().tolist(),
            'call_count': model.get_call_count(),
            'status': 'success'
        }
        
        # Вывод в UTF-8
        sys.stdout.buffer.write(json.dumps(result).encode('utf-8'))
        
    except Exception as e:
        error_result = {'error': str(e), 'status': 'failed'}
        sys.stdout.buffer.write(json.dumps(error_result).encode('utf-8'))
        sys.exit(1)

if __name__ == '__main__':
    main()