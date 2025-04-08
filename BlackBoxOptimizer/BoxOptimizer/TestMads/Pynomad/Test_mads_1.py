import PyNomad
import numpy as np
def bb(x):
    dim = x.size()
    f = sum([x.get_coord(i)**2 for i in range(dim)])    
    x.setBBO(str(f).encode("UTF-8"))
    return 1 # 1: success 0: failed evaluation

# Обязательные параметры (явно указываем float)
lb = [-10.0, -10.0, -10.0]  # Нижние границы
ub = [10.0, 10.0, 10.0]       # Верхние границы
x0 = [3.0, 2.0, 1.1]            # Начальная точка

# Доп. параметры оптимизации
params = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 100", "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ", "HISTORY_FILE history.txt", "MIN_MESH_SIZE 1e-6"]
result = PyNomad.optimize(bb, x0, lb, ub, params)
fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
output = "\n".join(fmt)
print("\nNOMAD results \n" + output + " \n")