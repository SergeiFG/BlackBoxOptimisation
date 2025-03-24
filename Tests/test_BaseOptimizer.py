"""
test_BaseOptimizer.py

Тестирование поведения методов класса BaseOptimizer

"""

import pytest

import numpy as np

from BlackBoxOptimizer.BoxOptimizer.BaseOptimizer import BaseOptimizer

def test__init__():
    pass


def test_getResult():
    loc_class = BaseOptimizer(1, 2, 5)
    with pytest.raises(NotImplementedError):
        loc_class.getResult()


def test__calc_objective_function_value():
    loc_class = BaseOptimizer(1, 2, 5)
    with pytest.raises(NotImplementedError):
        loc_class._calc_objective_function_value()
        

def test_vecToModel_getter_id():
    """Проверка: предоставляемый вектор для отправки в модель имеет отличный указатель на область 
    памяти от того, что хранится внутри класса"""
    loc_class = BaseOptimizer(1, 2, 5)
    assert id(loc_class.vecToModel) != id(loc_class._to_opt_model_data)


def test_vecToModel_setter():
    """Проверка: установка атрибута по сеттеру невозможна"""
    loc_class = BaseOptimizer(1, 2, 5)
    with pytest.raises(AttributeError):
        loc_class.vecToModel = 0


def test_vecFromModel_getter():
    """Проверка: чтение атрибута vecFromModel по геттеру невозможна"""
    loc_class = BaseOptimizer(1, 2, 5)
    with pytest.raises(AttributeError):
        loc_val = loc_class.vecFromModel


def test_vecFromModel_setter_None():
    """Проверка: установка атрибута vecFromModel new_value - None Ничего не должно измениться"""
    loc_class = BaseOptimizer(1, 2, 5)
    loc_mem_vec = loc_class._from_model_data.vec.copy()
    loc_class.vecFromModel = None
    assert loc_mem_vec.all() == loc_class._from_model_data.vec.all()


def test_vecFromModel_setter_another_type():
    """Проверка: установка атрибута vecFromModel new_value - неверного типа значения"""
    loc_class = BaseOptimizer(1, 2, 5)
    with pytest.raises(TypeError, match = "Неврный тип параметра для присовения атрибуту vecFromModel"):
        loc_class.vecFromModel = 15


def test_vecFromModel_setter_another_cell_type():
    """Проверка: передача массива неверного типа элементов"""
    loc_class = BaseOptimizer(1, 2, 5)
    loc_arr = np.array(np.zeros(2), dtype = int)
    with pytest.raises(TypeError, match = "Неверный тип элементов присваиваемого вектора"):
        loc_class.vecFromModel = loc_arr


def test_vecFromModel_setter_another_array_shape():
    """Проверка: передача массивов разного размера"""
    loc_class = BaseOptimizer(1, 2, 5)
    loc_arr = np.array(np.zeros(8), dtype = float)
    with pytest.raises(TypeError, match = "Неверная размерность присваемого вектора"):
        loc_class.vecFromModel = loc_arr


def test_vecFromModel_setter_correct_vec():
    """Проверка: подстановка корректнорго вектора"""
    loc_class = BaseOptimizer(1, 2, 5)
    loc_arr =  np.array(np.ones(2), dtype = float)
    loc_class.vecFromModel = loc_arr.copy()
    assert loc_class._from_model_data.vec.all() == loc_arr.all()


def test_vecFromModel_setter_correct_vec_diff_id():
    """Проверка: присваиваемый новый вектор отличный отдельный ID от внутреннего"""
    loc_class = BaseOptimizer(1, 2, 5)
    loc_arr =  np.array(np.ones(2), dtype = float)
    loc_class.vecFromModel = loc_arr
    assert id(loc_class._from_model_data.vec) != id(loc_arr)


def test_objFuncValue_getter():
    """Проверка работы геттера атрибута objFuncValue"""
    loc_class = BaseOptimizer(1, 2, 5)
    loc_class._objective_function_value_from_model = 154.33
    assert loc_class.objFuncValue == 154.33
    

def test_objFuncValue_setter_None():
    """Проверка: передан None - значение атрибута _objective_function_value_from_model не 
    изменилось"""
    loc_class = BaseOptimizer(1, 2, 5)
    loc_class._objective_function_value_from_model = 154.235
    loc_class.objFuncValue = None
    assert loc_class._objective_function_value_from_model == 154.235
   

def test_objFuncValue_setter_incotrrect_type():
    """Проверка: присвоение неверного типа"""
    loc_class = BaseOptimizer(1, 2, 5)
    with pytest.raises(TypeError, match = "Присваивается неверный тип атрибуту _objective_function_value_from_model"):
        loc_class.objFuncValue = str("1362434")


def test_objFuncValue_setter_cotrrect_types():
    """Проверка: присвоение верных типов параметров"""
    loc_class = BaseOptimizer(1, 2, 5)
    loc_class._objective_function_value_from_model = 285.333
    loc_class.objFuncValue = float(154.55)
    assert loc_class._objective_function_value_from_model == 154.55
    
    loc_class.objFuncValue = int(154)
    assert loc_class._objective_function_value_from_model == 154
    
