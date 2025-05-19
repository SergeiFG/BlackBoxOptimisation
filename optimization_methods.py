from BlackBoxOptimizer.BoxOptimizer.TestStepOpt.TestStepOpt import TestStepOpt
from BlackBoxOptimizer.BoxOptimizer.EvolutionaryOpt.EvolutionaryOpt import EvolutionaryOpt

OPTIMIZATION_METHODS = {
    "TestStepOpt": {
        "class": TestStepOpt,
        "default_params": {
            "step": 0.01
        }
    },
    "EvolutionaryOpt": {
        "class": EvolutionaryOpt,
        "default_params": {
            "population_size": 150,
            "offspring_per_parent": 5,
            "mutation_prob": 0.1,
            "sigma_init": 0.005,
            "t_max": 15000
        }
    }
}