from Models import DiscreteSquareSumModel

import numpy as np

if __name__ == "__main__":


    target_point = np.array([2, 0.5, -0.2, 1])  # Целевая точка, которую хотим увидеть, используется для отладки
    discrete_index = np.array([3])
    model = DiscreteSquareSumModel(target=target_point,discrete_indices=discrete_index)

    print(model.evaluate([2.1, 0.6, -0.1, 1]))
