from BlackBoxOptimizer.BoxOptimizer.GaussOpt.GaussOpt import GaussOpt
from BlackBoxOptimizer.BoxOptimizer.SimulatedAnnealingOptimizer.SimulatedAnnealingOptimizer import SimulatedAnnealingOptimizer
from BlackBoxOptimizer.BoxOptimizer.EvolutionaryOpt.EvolutionaryOpt import EvolutionaryOpt
from BlackBoxOptimizer.BoxOptimizer.TestStepOpt.TestStepOpt import TestStepOpt
from BlackBoxOptimizer.BoxOptimizer.Genetic_Algo.Genetic_Algo import Genetic_Algo

# Configuration for server-client communication
SERVER_HOST = 'localhost'
SERVER_PORT = 5085  # Default port, can be changed here

# Optimization methods configuration
OPTIMIZATION_METHODS = {
    "GaussOpt": {
        "class": GaussOpt,
        "default_params": {
            "kernel_cfg":('Matern', {'nu': 2.5})
        }
    },
    "SimulatedAnnealingOpt": {
        "class": SimulatedAnnealingOptimizer,
        "default_params": {
            "initial_temp": 100.0,
            "min_temp": 1e-8,
            "cooling_rate": 0.97,  # Более быстрое охлаждение
            "step_size": 0.5,      # Меньший шаг для более точного поиска
            "penalty_coef": 1e8    # Значительно увеличиваем штраф для строгого соблюдения ограничений
        }
    },
    "EvolutionaryOpt": {
        "class": EvolutionaryOpt,
        "default_params": {
            "population_size": 20,
            "offspring_per_parent": 5,
            "mutation_prob": 0.2,
            "sigma_init": 0.1
        }
    },
    "TestStepOpt": {
        "class": TestStepOpt,
        "default_params": {}
    },
    "Genetic_Algo": {
        "class": Genetic_Algo,
        "default_params": {
            "population_size": 50,
            "init_mutation": 0.2,
            "min_mutation": 0.01,
            "elite_size": 10
        }
    }
}