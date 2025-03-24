"""
test_OptimizedVectorData.py

Тестирование поведения методов класса OptimizedVectorData

"""


import pytest

import numpy as np

from BlackBoxOptimizer.BoxOptimizer.BaseOptimizer import OptimizedVectorData


def test_on_init():
    """Проверка при инициалищзации объекта класса"""
    loc_class = OptimizedVectorData(size = 4)
    assert loc_class._vec[0][OptimizedVectorData.min_index] == -np.inf
    assert loc_class._vec[0][OptimizedVectorData.max_index] ==  np.inf
    assert loc_class._vec[1][OptimizedVectorData.min_index] == -np.inf
    assert loc_class._vec[1][OptimizedVectorData.max_index] ==  np.inf
    assert loc_class._vec[2][OptimizedVectorData.min_index] == -np.inf
    assert loc_class._vec[2][OptimizedVectorData.max_index] ==  np.inf
    assert loc_class._vec[3][OptimizedVectorData.min_index] == -np.inf
    assert loc_class._vec[3][OptimizedVectorData.max_index] ==  np.inf


def test_setLimitation_on_None():
    """Проверка установки значений функции оба None"""
    loc_class = OptimizedVectorData(size = 4)
    loc_class.setLimitation(0)
    assert loc_class._vec[0][OptimizedVectorData.min_index] == -np.inf
    assert loc_class._vec[0][OptimizedVectorData.max_index] ==  np.inf


def test_on_BOTH():
    """Проверка установлены оба значения"""
    loc_class = OptimizedVectorData(size = 4)
    loc_class.setLimitation(1, min = 8, max = 23)
    assert loc_class._vec[1][OptimizedVectorData.min_index] == 8
    assert loc_class._vec[1][OptimizedVectorData.max_index] == 23


def test_on_Min_G():
    """Проврка: Вызов ошибки при минимуме больше максимума"""
    loc_class = OptimizedVectorData(size = 4)
    with pytest.raises(ValueError):
        loc_class.setLimitation(1, min = 23, max = 8)


def test_on_Min():
    """Проверка: установка только минимума в первый раз"""
    loc_class = OptimizedVectorData(size = 4)
    loc_class.setLimitation(1, min = 23)
    assert loc_class._vec[1][OptimizedVectorData.min_index] == 23
    assert loc_class._vec[1][OptimizedVectorData.max_index] == np.inf


def test_on_Min_set():
    """Проверка: установлен максимум, ставим минимум, смотрим, что максимум без изменений"""
    loc_class = OptimizedVectorData(size = 4)
    loc_class._vec[1][OptimizedVectorData.max_index] = 30
    loc_class.setLimitation(1, min = 23)
    assert loc_class._vec[1][OptimizedVectorData.min_index] == 23
    assert loc_class._vec[1][OptimizedVectorData.max_index] == 30


def test_on_Min_set_G():
    """Проверка: установлен максимум, ставим минимум, превосходящий максимум, ожидаем ошибку"""
    loc_class = OptimizedVectorData(size = 4)
    loc_class._vec[1][OptimizedVectorData.max_index] = 10
    with pytest.raises(ValueError):
        loc_class.setLimitation(1, min = 23)


def test_on_Max():
    """Проверка: установка только максимума в первый раз"""
    loc_class = OptimizedVectorData(size = 4)
    loc_class.setLimitation(1, max = 23)
    assert loc_class._vec[1][OptimizedVectorData.min_index] == -np.inf
    assert loc_class._vec[1][OptimizedVectorData.max_index] ==  23


def test_on_Max_set():
    """Проверка: установлен минимум, ставим максимум, смотрим, что минимум без изменений"""
    loc_class = OptimizedVectorData(size = 4)
    loc_class._vec[1][OptimizedVectorData.min_index] = -30
    loc_class.setLimitation(1, max = 23)
    assert loc_class._vec[1][OptimizedVectorData.min_index] == -30
    assert loc_class._vec[1][OptimizedVectorData.max_index] ==  23


def tes_on_Max_set_L():
    """Проверка: установлен минимум, ставим максимум, который меньше минимума, ожидаем ошбку"""
    loc_class = OptimizedVectorData(size = 4)
    loc_class._vec[1][OptimizedVectorData.min_index] = 10
    with pytest.raises(ValueError):
        loc_class.setLimitation(1, max = -23)
        
    
def test_setVectorRandVal():
    """Проверка: Получение вектора с величинами по нормальному распределению"""
    loc_class = OptimizedVectorData(size = 4)
    loc_class.setVectorRandVal(min_val = 0.0, max_val = 1.0)
    for i in range(4):
        assert loc_class.vec[i] >= 0.0 and loc_class.vec[i] <= 1.0
        