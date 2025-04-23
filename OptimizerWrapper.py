#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import numpy as np
from BlackBoxOptimizer import TestStepOpt, EvolutionaryOpt, Optimizer
from Example import ModelMinSquareSum

METHOD_CONFIGS = {
    "GradientDescent": {
        "opt_class": TestStepOpt,
        "params": {"step": 0.01, "seed": 1546},
        "eval_mode": "full"
    },
    "Evolutionary": {
        "opt_class": EvolutionaryOpt,
        "params": {
            "population_size": 50,  # Увеличено с 30
            "offspring_per_parent": 2,
            "mutation_prob": 0.2,   # Уменьшено с 0.3
            "sigma_init": 0.1,      # Уменьшено с 0.2
            "seed": 42,
            "t_max": 50,
            "restarts": 3           # Добавлены рестарты
        },
        "eval_mode": "first"
    }
}

def safe_evolutionary_run(opt, model, config, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            opt.modelOptimize(func=lambda x: model.evaluate(x)[0])
            return opt.getOptimizer()
        except np.linalg.LinAlgError:
            if attempt < max_attempts - 1:
                # Увеличиваем разнообразие при рестарте
                opt.configure(sigma_init=config['params']['sigma_init'] * 1.5)
                continue
            raise

def run_optimization(params):
    try:
        method = params['method']
        config = METHOD_CONFIGS[method]
        
        # Автоподстройка параметров для Evolutionary
        if method == "Evolutionary":
            dim = params['to_model_vec_size']
            config['params']['population_size'] = max(50, dim * 10)
        
        # Инициализация
        target = np.zeros(params['to_model_vec_size'])
        model = ModelMinSquareSum(target)
        
        # Создание оптимизатора
        opt = Optimizer(
            optCls=config['opt_class'],
            seed=config['params']['seed'],
            to_model_vec_size=params['to_model_vec_size'],
            from_model_vec_size=2,
            iter_limit=params['iter_limit'],
            **({'dimension': params['to_model_vec_size']} if method == "Evolutionary" else {})
        )

        # Конфигурация
        if method == "GradientDescent":
            opt.configure(step=config['params']['step'])
            def evaluate(x): return model.evaluate(x)
        else:
            for i in range(params['to_model_vec_size']):
                opt.setVecItemLimit(i, "to_model", min=-1.0, max=1.0)
            opt.configure(**{k: v for k, v in config['params'].items() 
                         if k not in ['seed', 't_max', 'restarts']})

        # Запуск
        if method == "GradientDescent":
            opt.modelOptimize(func=evaluate)
            optimizer = opt.getOptimizer()
            result = optimizer.getResult()
        else:
            optimizer = safe_evolutionary_run(opt, model, config)
            result = optimizer._to_opt_model_data.vecs[:, 0]

        # Форматирование результата
        output = {
            "optimized_values": np.array(result).tolist(),
            "call_count": model.get_call_count(),
            "iterations_used": params['iter_limit'],
            "status": "success"
        }
        print("FINAL_RESULT:" + json.dumps(output))

    except Exception as e:
        print("ERROR:" + json.dumps({
            "error": str(e),
            "status": "failed",
            "method": method,
            "attempt": getattr(optimizer, 'restart_attempt', 0) if 'optimizer' in locals() else 0
        }))
        exit(1)

if __name__ == "__main__":
    try:
        run_optimization(json.loads(input()))
    except Exception as e:
        print("CRITICAL_ERROR:" + str(e))
        exit(1)