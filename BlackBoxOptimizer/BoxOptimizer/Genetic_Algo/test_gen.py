import numpy as np
from OptClass import GeneticAlgorithmOptimizer  # Импорт вашего класса
from typing import List

class OptimizationModel:
    def __init__(self):
        # Текущие значения MV (центры допустимых диапазонов)
        self.curr_mv1 = (9.4 + 17.2) / 2
        self.curr_mv2 = (12.3 + 22.6) / 2
        self.curr_mv3 = (28 + 40) / 2
        self.curr_mv4 = (94 + 101) / 2
        self.curr_mv5 = (0.0000001 + 2.8) / 2
        self.curr_mv6 = (0.0000001 + 1.3) / 2

        # Текущие значения CV (центры допустимых диапазонов)
        self.curr_cv1 = (9 + 16.5) / 2
        self.curr_cv2 = (9.4 + 17.2) / 2
        self.curr_cv3 = (81.5 + 88.5) / 2
        self.curr_cv4 = (89 + 91) / 2
        self.curr_cv5 = (12.3 + 22.6) / 2
        self.curr_cv6 = (85 + 140) / 2
        self.curr_cv7 = (19.6 + 36) / 2
        self.curr_cv8 = (28 + 40) / 2
        self.curr_cv9 = (0.28 + 1.16) / 2
        self.curr_cv10 = (0.85 + 1) / 2
        self.curr_cv11 = (10 + 25) / 2
        self.curr_cv12 = (95 + 99) / 2
        self.curr_cv13 = (33.3 + 59.9) / 2

    def evaluate(self, mv_values: list) -> list:
        x1, x2, x3, x4, x5, x6 = mv_values
        
        # Квадратичная целевая функция с минимумом в центре
        target_func = ((x1 - 13.3)**2 + (x2 - 17.45)**2 + 
                       (x3 - 34)**2 + (x4 - 97.5)**2 + 
                       (x5 - 1.4)**2 + (x6 - 0.65)**2)
        
        # Линейные модели с гарантированно совместными ограничениями
        additional_cv = [
            self.curr_cv1 + 0.3*(x1 - self.curr_mv1),  # CV1
            self.curr_cv2 - 0.2*(x1 - self.curr_mv1) + 0.1*(x2 - self.curr_mv2),  # CV2
            self.curr_cv3 + 0.15*(x3 - self.curr_mv3),  # CV3
            self.curr_cv4 - 0.05*(x4 - self.curr_mv4),  # CV4
            self.curr_cv5 + 0.25*(x2 - self.curr_mv2),  # CV5
            self.curr_cv6 - 0.1*(x2 - self.curr_mv2) + 0.1*(x3 - self.curr_mv3),  # CV6
            self.curr_cv7 + 0.2*(x3 - self.curr_mv3) - 0.1*(x4 - self.curr_mv4),  # CV7
            self.curr_cv8 + 0.15*(x3 - self.curr_mv3),  # CV8
            self.curr_cv9 + 0.02*(x3 - self.curr_mv3) + 0.005*(x4 - self.curr_mv4),  # CV9
            self.curr_cv10 - 0.03*(x1 - self.curr_mv1) - 0.02*(x2 - self.curr_mv2) + 
                0.01*(x3 - self.curr_mv3) + 0.02*(x4 - self.curr_mv4) + 
                0.01*(x5 - self.curr_mv5) - 0.01*(x6 - self.curr_mv6),  # CV10
            self.curr_cv11 - 0.05*(x1 - self.curr_mv1) - 0.03*(x2 - self.curr_mv2) + 
                0.04*(x3 - self.curr_mv3) + 0.03*(x4 - self.curr_mv4) + 
                0.02*(x5 - self.curr_mv5) - 0.01*(x6 - self.curr_mv6),  # CV11
            self.curr_cv12 - 0.02*(x1 - self.curr_mv1) - 0.03*(x2 - self.curr_mv2) + 
                0.01*(x3 - self.curr_mv3) + 0.02*(x4 - self.curr_mv4) + 
                0.01*(x5 - self.curr_mv5) - 0.01*(x6 - self.curr_mv6),  # CV12
            self.curr_cv13 + 0.1*(x1 - self.curr_mv1) + 
                0.08*(x2 - self.curr_mv2) + 0.12*(x3 - self.curr_mv3)  # CV13
        ]
        
        return [target_func] + additional_cv
    
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

# Границы для MV (управляющих переменных)
mv_bounds = {
    'lower': np.array([9.4, 12.3, 28, 94, 0.0000001, 0.0000001]),
    'upper': np.array([17.2, 22.6, 40, 101, 2.8, 1.3])
}

# Границы для CV (выходных параметров)
cv_bounds = {
    'lower': np.array([
        -np.inf,
        9,      # CV1
        9.4,    # CV2
        81.5,   # CV3
        89,     # CV4
        12.3,   # CV5
        85,     # CV6
        19.6,   # CV7
        28,     # CV8
        0.28,   # CV9
        0.85,   # CV10
        10,     # CV11
        95,     # CV12
        33.3    # CV13
    ]),
    'upper': np.array([
        np.inf,
        16.5,   # CV1
        17.2,   # CV2
        88.5,   # CV3
        91,     # CV4
        22.6,   # CV5
        140,    # CV6
        36,     # CV7
        40,     # CV8
        1.16,   # CV9
        1,      # CV10
        25,     # CV11
        99,     # CV12
        59.9    # CV13
    ])
}

# Инициализация модели
model = OptimizationModel()

# Функция для оптимизации (обертка для GA)
def optimization_function(x):
    # Применяем границы MV перед расчетом
    x_clipped = np.clip(x, mv_bounds['lower'], mv_bounds['upper'])
    result = model.evaluate(x_clipped)
    return result

# Параметры оптимизации
dimension = 6  # 6 MV
population_size = 800
generations = 250
elite_size = 15

# Инициализация оптимизатора
optimizer = GeneticAlgorithmOptimizer(
    func=optimization_function,
    dimension=dimension,
    population_size=population_size,
    generations=generations,
    elite_size=elite_size,
    lower_bounds=mv_bounds['lower'],
    upper_bounds=mv_bounds['upper'],
    output_lower_bounds=cv_bounds['lower'],
    output_upper_bounds=cv_bounds['upper'],
    init_mutation=0.5,
    min_mutation=0.0005,
)

# Запуск оптимизации
best_solution, best_fitness, best_output = optimizer.run()

# Вывод результатов
print("\nРезультаты оптимизации:")
print(f"Лучшее решение (MV): {best_solution}")
print(f"Лучшая приспособленность: {best_fitness:.2f}")
print(f"Целевая функция (прибыль): {best_output[0]:.2f}")

print("\nПроверка ограничений CV:")
for i in range(14):
    cv_value = best_output[i]
    lb = cv_bounds['lower'][i]
    ub = cv_bounds['upper'][i]
    status = "✅" if lb <= cv_value <= ub else "❌"
    print(f"CV{i}: {cv_value:.6f} [{lb}, {ub}] {status}")

# Проверка выполнения всех ограничений
all_constraints_satisfied = all(
    cv_bounds['lower'][i] <= best_output[i] <= cv_bounds['upper'][i]
    for i in range(14)
)

if all_constraints_satisfied:
    print("\n🎉 ВСЕ ОГРАНИЧЕНИЯ ВЫПОЛНЕНЫ!")
else:
    print("\n⚠️ НЕ ВСЕ ОГРАНИЧЕНИЯ ВЫПОЛНЕНЫ!")
