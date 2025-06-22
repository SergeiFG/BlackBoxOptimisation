from BlackBoxOptimizer import Optimizer, EvolutionaryOpt, SimulatedAnnealingOptimizer, GaussOpt
from Models import SquareSumModel, SinParabolaModel

from typing import List

import numpy as np

import random
import statistics
from math import sqrt
import math

class OptimizationModel:
    def __init__(self, a=1.0):
        self.a = a

        self.curr_mv1=14.1
        self.curr_mv2=15.7
        self.curr_mv3=34.7
        self.curr_mv4=98.7
        self.curr_mv5=0.93
        self.curr_mv6=0.43

        self.curr_cv1=13.5
        self.curr_cv2=14.1
        self.curr_cv3=83.5
        self.curr_cv4=90.3
        self.curr_cv5=15.7
        self.curr_cv6=100
        self.curr_cv7=30.5
        self.curr_cv8=34.7
        self.curr_cv9=1.145
        self.curr_cv10=0.95
        self.curr_cv11=20
        self.curr_cv12=95
        self.curr_cv13=61.1


    def evaluate(self, mv_values: List[float]) -> float:
        # Минимум: x1=2, x2=-3, x3=4, x4=1, значение 0
        x1, x2, x3, x4, x5, x6 = mv_values
        #target_func=(40000*additional_cv[13-1]-65000*x5-45000*x6)
        additional_cv = [
            self.curr_cv1+0.964*(x1-self.curr_mv1),  # CV1 (индекс 1)
            self.curr_cv2+1*(x1-self.curr_mv1),   # CV2 (индекс 2)
            self.curr_cv3+1.4*(x1-self.curr_mv1), # CV3 (индекс 3)
            self.curr_cv4-0.4*(x1-self.curr_mv1), # CV4 (индекс 4)
            self.curr_cv5+1*(x2-self.curr_mv2),   # CV5 (индекс 5)
            self.curr_cv6-0.5*(x2-self.curr_mv2)+0.5*(x3-self.curr_mv3),               # CV6 (индекс 6)
            self.curr_cv7+0.88*(x3-self.curr_mv3)-2.65*(x4-self.curr_mv4),              # CV7 (индекс 7)
            self.curr_cv8+1*(x3-self.curr_mv3),               # CV8 (индекс 8)
            self.curr_cv9+0.033*(x3-self.curr_mv3)+0.01*(x4-self.curr_mv4),       # CV9 (индекс 9)
            self.curr_cv10-0.0218*(x1-self.curr_mv1)-0.015*(x2-self.curr_mv2)+0.0288*(x3-self.curr_mv3)+0.109*(x4-self.curr_mv4)+0.0655*(x5-self.curr_mv5)-0.0218*(x6-self.curr_mv6),             # CV10 (индекс 10)
            self.curr_cv11-0.546*(x1-self.curr_mv1)-0.382*(x2-self.curr_mv2)+0.721*(x3-self.curr_mv3)+0.719*(x4-self.curr_mv4)+0.764*(x5-self.curr_mv5)-0.546*(x6-self.curr_mv6),                # CV11 (индекс 11)
            self.curr_cv12-0.109*(x1-self.curr_mv1)-0.546*(x2-self.curr_mv2)+0.0546*(x3-self.curr_mv3)+0.625*(x4-self.curr_mv4)+0.371*(x5-self.curr_mv5)-0.109*(x6-self.curr_mv6),        # CV12 (индекс 12)
            self.curr_cv13+0.962*(x1-self.curr_mv1)+1*(x2-self.curr_mv2)+0.88*(x3-self.curr_mv3)         # CV13 (индекс 13)
        ]
        target_func=(40000*additional_cv[13-1]-65000*x5-45000*x6)
        return [target_func]+additional_cv

cvs = [
            {
                "Id": "36127bf6-bf83-45c0-a4e1-65d2a1c20c22",
                "Name": "Target Function",
                "DataType": "Numeric",
                "LowerBound": -np.inf,
                "UpperBound": np.inf
            },
            {
                "Id": "cv1", "Name": "Y1", "DataType": "Numeric", "LowerBound": 9,"UpperBound": 16.5
            },
            {
                "Id": "cv2", "Name": "Y2", "DataType": "Numeric", "LowerBound": 9.4,"UpperBound": 17.2
            },
            {
                "Id": "cv3", "Name": "Y3", "DataType": "Numeric", "LowerBound": 81.5,"UpperBound": 88.5
            },
            {
                "Id": "cv4", "Name": "Y4", "DataType": "Numeric", "LowerBound": 89,"UpperBound": 91
            },
            {
                "Id": "cv5", "Name": "Y5", "DataType": "Numeric", "LowerBound": 12.3,"UpperBound": 22.6
            },
            {
                "Id": "cv6", "Name": "Y6", "DataType": "Numeric", "LowerBound": 85,"UpperBound": 140
            },
            {
                "Id": "cv7", "Name": "Y7", "DataType": "Numeric", "LowerBound": 19.6,"UpperBound": 36
            },
            {
                "Id": "cv8", "Name": "Y8", "DataType": "Numeric", "LowerBound": 28,"UpperBound": 40
            },
            {
                "Id": "cv9", "Name": "Y9", "DataType": "Numeric", "LowerBound": 0.28,"UpperBound": 1.16
            },
            {
                "Id": "cv10", "Name": "Y10", "DataType": "Numeric", "LowerBound": 0.85,"UpperBound": 1
            },
            {
                "Id": "cv11", "Name": "Y11", "DataType": "Numeric", "LowerBound": 10,"UpperBound": 25
            },
            {
                "Id": "cv12", "Name": "Y12", "DataType": "Numeric", "LowerBound": 95,"UpperBound": 99
            },
            {
                "Id": "cv13", "Name": "Y13", "DataType": "Numeric", "LowerBound": 33.3,"UpperBound": 59.9
            }
        ]

mvs = [
    {"Id": "mv1", "Name": "X1", "DataType": "Numeric", "LowerBound": 9.4, "UpperBound": 17.2},
    {"Id": "mv2", "Name": "X2", "DataType": "Numeric", "LowerBound": 12.3, "UpperBound": 22.6},
    {"Id": "mv3", "Name": "X3", "DataType": "Numeric", "LowerBound": 28, "UpperBound": 40},
    {"Id": "mv4", "Name": "X4", "DataType": "Numeric", "LowerBound": 94, "UpperBound": 101},
    {"Id": "mv5", "Name": "X5", "DataType": "Numeric", "LowerBound": 0.0000001, "UpperBound": 2.8},
    {"Id": "mv6", "Name": "X6", "DataType": "Numeric", "LowerBound": 0.0000001, "UpperBound": 1.3}
]

# Минимум: x1=2, x2=-3, x3=4, x4=1, значение 0
model = OptimizationModel()
model.evaluate([2.0, -3.0, 4.0, 1.0, 0.0000001, 0.0000001])

import warnings

warnings.filterwarnings('ignore')


optimizer = Optimizer(
        optCls              = GaussOpt,
        seed                = 15, # TODO: Проверить, точно ли работает. Сейчас выдаёт разные значения при одном seed
        to_model_vec_size   = 6,
        from_model_vec_size = 14,
        iter_limit          = 50,
        external_model = model.evaluate,
        target = None,
        )
# optimizer.configure(kernel_cfg=('RBF',{}))

for i in range(6):  # Первые 3 параметра - непрерывные
    optimizer.setVecItemLimit(i, "to_model", min=mvs[i]['LowerBound'], max=mvs[i]['UpperBound'])

for i in range(1, 14):  # Первые 3 параметра - непрерывные
    optimizer.setVecItemLimit(i, "from_model", min=cvs[i]['LowerBound'], max=cvs[i]['UpperBound'])
# optimizer.setVecItemLimit(1, "from_model", min=1, max=np.inf)


optimizer.modelOptimize()


currentOptimizer = optimizer.getOptimizer()

x_hist = currentOptimizer.history_to_opt_model_data.copy()

x = optimizer.getResult()
print(' Итог MV', x)
print(' Итог СV',model.evaluate(x))
print(currentOptimizer.most_opt_vec)
#print(' Итог L1', np.sum(np.abs(x-target)))
print(' число обращений к оракулу', optimizer.get_usage_count())
