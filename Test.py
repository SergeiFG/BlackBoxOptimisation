import numpy as np
from BlackBoxOptimizer import SimulatedAnnealingOptimizer, Optimizer

def quadratic(x):
    """
    Квадратичная функция (параболоид)
    Минимум: [0, 0, ..., 0] со значением 0
    Рекомендуемые параметры:
    - bounds: [-10, 10] для всех переменных
    - initial_temperature: 100
    - cooling_rate: 0.95
    """
    return np.array([np.sum(x**2)])

def rastrigin(x):
    """
    Функция Растригина (многомодальная с множеством локальных минимумов)
    Минимум: [0, 0, ..., 0] со значением 0
    Рекомендуемые параметры:
    - bounds: [-5.12, 5.12] (классический вариант)
    - initial_temperature: 200 (нужна высокая для выхода из локальных минимумов)
    - cooling_rate: 0.99 (медленное охлаждение)
    """
    A = 10
    n = len(x)
    return np.array([A*n + np.sum(x**2 - A*np.cos(2*np.pi*x))])

def rosenbrock(x):
    """
    Функция Розенброка (долина с плоским дном)
    Минимум: [1, 1, ..., 1] со значением 0
    Рекомендуемые параметры:
    - bounds: [-5, 10] (асимметричные границы)
    - initial_temperature: 1000 (большая для длинной долины)
    - cooling_rate: 0.999 (очень медленное охлаждение)
    """
    return np.array([np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)])

def ackley(x):
    """
    Функция Экли (сложный ландшафт с почти плоскими областями)
    Минимум: [0, 0, ..., 0] со значением 0
    Рекомендуемые параметры:
    - bounds: [-32.768, 32.768] (классический вариант)
    - initial_temperature: 50 (достаточно низкая из-за плоских областей)
    - cooling_rate: 0.9
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2*np.pi*x))
    return np.array([-20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e])

def himmelblau(x):
    """
    Функция Химмельблау (4 равных глобальных минимума в 2D)
    Минимумы:
    (3.0, 2.0), (-2.805118, 3.131312), 
    (-3.779310, -3.283186), (3.584428, -1.848126)
    Все со значением 0
    Рекомендуемые параметры:
    - bounds: [-5, 5] для обеих переменных
    - initial_temperature: 100
    - cooling_rate: 0.95
    """
    return np.array([(x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2])

FUNCTIONS = {
    '1': ('Квадратичная функция', quadratic, 3, (-10, 10), 100, 0.95),
    '2': ('Функция Растригина', rastrigin, 3, (-5.12, 5.12), 200, 0.99),
    '3': ('Функция Розенброка', rosenbrock, 3, (-5, 10), 1000, 0.999),
    '4': ('Функция Экли', ackley, 3, (-32.768, 32.768), 50, 0.9),
    '5': ('Функция Химмельблау', himmelblau, 2, (-5, 5), 100, 0.95)
}

print("Доступные тестовые функции:")
for key, (name, _, _, _, _, _) in FUNCTIONS.items():
    print(f"{key}: {name}")

selected = input("Выберите функцию для оптимизации (1-5): ")
while selected not in FUNCTIONS:
    print("Некорректный ввод!")
    selected = input("Выберите функцию для оптимизации (1-5): ")

func_name, func, dim, bounds_range, init_temp, cool_rate = FUNCTIONS[selected]
bounds = [bounds_range for _ in range(dim)]

mode = input("Оптимизировать (1) минимум или (2) максимум? [1/2]: ")
maximize = mode == '2'

optimizer = Optimizer(
    optCls=SimulatedAnnealingOptimizer,
    to_model_vec_size=dim,
    from_model_vec_size=1,
    iter_limit=1000,
    initial_temperature=init_temp,
    cooling_rate=cool_rate,
    maximize=maximize
)

for i in range(dim):
    optimizer.setVecItemLimit(
        index=i, 
        vec_dir="to_model", 
        min=bounds[i][0], 
        max=bounds[i][1]
    )

print(f"\nНачинаем оптимизацию функции {func_name}...")
print(f"Рекомендуемые параметры:")
print(f"- Размерность: {dim}")
print(f"- Границы: {bounds}")
print(f"- Начальная температура: {init_temp}")
print(f"- Скорость охлаждения: {cool_rate}")
print(f"- Режим: {'максимум' if maximize else 'минимум'}")

optimizer.modelOptimize(func)

# Получение результатов
best_params, best_value = optimizer.getResult()
history = optimizer.getHistoricalData("obj_val")

print(f"\nРезультаты оптимизации:")
print(f"Найденные параметры: {best_params}")
print(f"Значение функции: {best_value}")
print(f"Ожидаемый {'максимум' if maximize else 'минимум'}: 0 (для большинства функций)")