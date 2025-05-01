"""
BaseOptimizer
---

Файл базового оптимизатора

Определяет общие методы и атрибуты для всех частных оптимизаторов

"""

# Подключаемые модули независимой конфигурации
from typing import TypeVar, Callable, Tuple, Literal
import numpy as np
import time
import random



# Подключаемые модули зависимой конфигурации``
if __name__ == "__main__":
    pass
else:
    pass


DEBUG_THE_BIGGEST_ONE = 999999999
"""Самое большое число. Пока в дебаге"""



class _vectorItemProperties(object):
    """Базовый класс типа элемента вектора занчений
    
    """
    def __init__(self, 
                min : float | int,
                max : float | int
                ) -> None:

        if min is not None and max is not None and min >= max:
            raise ValueError(f"Значение {min} не может быть больше {max}")
        
        self._min  : float | int = min
        self._max  : float | int = max

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, new_val):
        if new_val is None:
            return
        if not isinstance(new_val, (float, int)):
            return
        self._min = new_val

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, new_val):
        if new_val is None:
            return
        if not isinstance(new_val, (float, int)):
            return
        self._max = new_val


    def randCreate():
        """
        randCreate
        ----
        Метод получения допустимого рандомного числа
        """
        raise NotImplementedError


    def isCorrect(value) -> bool:
        """
        isCorrect
        ---
        Метод проверки корректности значения
        """
        raise NotImplementedError


    def isWitihRange(self, value) -> bool:
        """Проверка нахождения в диапазоне допустимых значений"""
        return self._min <= value and value <= self._max

    def __str__(self):
        return f"<{self.__class__.__name__} : min [{self._min}] max [{self._max}]>"



class _floatItem(_vectorItemProperties):
    """"Элементы типа FLOAT"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def randCreate(self) -> float:
        return random.uniform(
            a = self._min if self._min != -np.inf else -DEBUG_THE_BIGGEST_ONE,
            b = self._max if self._max !=  np.inf else  DEBUG_THE_BIGGEST_ONE
            )


    def isCorrect(self, value) -> bool:
        """
        isCorrect
        ---
        Метод проверки корректности значения
        """
        if not isinstance(value, (float, int)):
            return False
        if not self.isWitihRange(value): 
            return False
        return True



class _boolItem(_vectorItemProperties):
    """"Элементы типа BOOL"""
    def __init__(self):
        super().__init__(min = 0, max = 1)

    def randCreate(self) -> float:
        return random.randint(0, 1)

    def isCorrect(self, value) -> bool:
        """
        isCorrect
        ---
        Метод проверки корректности значения
        """
        if not isinstance(value, (float, int)):
            return False
        if not self.isWitihRange(value): 
            return False
        if value != 0  or value != 1: 
            return False
        return True

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, new_val):
        self._min = 0

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, new_val):
        self._max = 1



class OptimizedVectorData(object):
    
    # Перечисление индексов во внутреннем массиве класса
    # min_index          : int = 0
    # max_index          : int = 1
    values_index_start : int = 0
    
    # Положение векторов
    axis_X : int = 0
    axis_Y : int = 1

    def __init__(
        self,
        vec_size : int,
        seed     : int,
        vec_candidates_size : int = 1
        ) -> None:
        """
        
        Аргументы:
            vec_size            : int - Размерность вектора
            vec_candidates_size : int - Количество векторов кандидатов (по умолчанию 1)
        """
        # Корректность атрибута vec_size
        if vec_size is None:
            raise ValueError("Значение параметра vec_size не может быть None")
        if not isinstance(vec_size, int):
            raise TypeError("Передан неверный тип параметра vec_size")
        if vec_size <= 0:
            raise ValueError("Значение размера vec_size не может быть меньше либо равно 0")

        # Корректность атрибута vec_candidates_size
        if vec_candidates_size is None:
            raise ValueError("Значение параметра vec_candidates_size не может быть None")
        if not isinstance(vec_candidates_size, int):
            raise TypeError("Передан неверный тип параметра vec_candidates_size")
        if vec_candidates_size <= 0:
            raise ValueError("Значение размера vec_candidates_size не может быть меньше либо равно 0")


        self._seed : int = seed
        """Используемая база генератора для псевдослучайных последовательностей"""


        self._vec_size : int = vec_size
        """Размер хранимого вектора"""
        self._vec_candidates_size : int = vec_candidates_size

        self._vec : np.array = np.array(
            [
                [0.0 for II in range(self._vec_candidates_size + OptimizedVectorData.values_index_start)] \
                    for I in range(self._vec_size)
                ],
            dtype = float
        )
        # self._vec[:, OptimizedVectorData.min_index] = -np.inf
        # self._vec[:, OptimizedVectorData.max_index] =  np.inf
        """Хранимый вектор значений, инициализируемый нулями, хранит значение минимума и максимума"""
        
        
        self._values_properties_list : list[_floatItem | _boolItem] = \
            [_floatItem(-np.inf, np.inf) for _ in range(self._vec_size)]
        """Вектор хранения свойств элементов"""

        self._values_allowed_types : list = [cls.__name__ for cls in _vectorItemProperties.__subclasses__()]
        

    def DEBUG_printInfo(self) -> None:
        """
        Отладочная печать содержимого
        """
        np.set_printoptions(linewidth = 150)
        loc_max_c_property : int = 0
        loc_max_c_index    : int = 8
        loc_max_c_values   : int = 0
        for item_prop, index, vec in zip(self._values_properties_list, range(self._vec_size), self._vec[:]):
            loc_max_c_property = len(str(item_prop)) if len(str(item_prop)) > loc_max_c_property else loc_max_c_property
            loc_max_c_index    = len(str(index))     if len(str(index))     > loc_max_c_index    else loc_max_c_index
            loc_max_c_values   = len(str(vec))       if len(str(vec))       > loc_max_c_values   else loc_max_c_values
        
        print("| {property:{loc_max_c_property}} | {index:{loc_max_c_index}} | {values:{loc_max_c_values}} |".format(
            clt_b = '\033[43m',
            property = "Тип элементов",
            index    = "Индекс",
            values   = "Значения",
            loc_max_c_property = loc_max_c_property,
            loc_max_c_index    = loc_max_c_index,
            loc_max_c_values   = loc_max_c_values
        ))
        print("|" + "-"*(loc_max_c_property + 2) + "|" + "-"*(loc_max_c_index + 2) + "|" + "-"*(loc_max_c_values + 2) + "|")
        for item_prop, index, vec in zip(self._values_properties_list, range(self._vec_size), self._vec[:]):
            print("| {property:{loc_max_c_property}} | {index:{loc_max_c_index}} | {values:{loc_max_c_values}} |".format(
                property = str(item_prop),
                index    = index,
                values   = str(vec),
                loc_max_c_property = loc_max_c_property,
                loc_max_c_index    = loc_max_c_index,
                loc_max_c_values   = loc_max_c_values
            ))
        np.set_printoptions()


    @property
    def vecs(self) -> np.ndarray:
        return np.copy(self._vec[:, OptimizedVectorData.values_index_start:])


    def iterVectors(self) -> np.array:
        """
        iterVectors
        ---
        Итератор входящих вектров, без лимитирующих векторов
        """
        # TODO: Подумать над управлением доступом
        for column in range( \
            np.size(self._vec[OptimizedVectorData.axis_Y]) - OptimizedVectorData.values_index_start):
            yield self._vec[:, column + OptimizedVectorData.values_index_start]


    def setVecItemType(self, index : int, new_type : Literal["float", "bool"], *args, **kwargs) -> None:
        """
        setVecItemType
        ---
        Установка типа элемента вектора
        
        index    : int - Индекс изменяемого элемента вектора
        new_type : Literal["float", "bool"]
        """
        loc_class_name = f"_{new_type}Item"
        if loc_class_name not in self._values_allowed_types:
            raise KeyError(f"Невозможно присвоить элементу тип {new_type}")
        self._values_properties_list[index] = eval(loc_class_name)(*args, **kwargs)


    def setLimitation(self, 
                      index : int, 
                      min : None | float = None,
                      max : None | float = None) -> None:
        """
        setLimitation
        ---
        
        Установка ограничения для параметра внутри вектора
        """
        if min is not None and max is not None and min >= max:
            raise ValueError(f"Значение {min} не может быть больше {max}")

        self._values_properties_list[index].min = min
        self._values_properties_list[index].max = max



    def setVectorRandValByLimits(self) -> None:
        """
        setVectorRandVal
        ---
        TODO: Получение начального вектора с величинами в диапазонах допустимого минимума и максимума 
        по нормальному распределению 
        """
        random.seed(int(self._seed))

        for vec, index in zip(
            self._vec[:, OptimizedVectorData.values_index_start:], 
            range(self._vec_size)
            ):
            for i in range(np.size(vec)):
                vec[i] = self._values_properties_list[index].randCreate()



    def getInLimitsMatrix(self) -> np.array:
        """
        getInLimitsMatrix
        ---
        Получение бинарной матрицы признаков принадлежности параметра вектора диапазону
        минимум-максимум
        """
        loc_matrix = np.array(
            np.zeros(
                shape = np.shape(
                    self._vec[:,OptimizedVectorData.values_index_start:])), dtype = bool)
        
        for vec, bool_vec, index in zip(
            self._vec[:, OptimizedVectorData.values_index_start:], 
            loc_matrix,
            range(self._vec_size)
            ):
            for vec_item_val, bool_item_num in zip(vec, range(np.size(bool_vec))):
                bool_vec[bool_item_num] = \
                    self._values_properties_list[index].isWitihRange(vec_item_val)
                    
        return loc_matrix


    def __str__(self):
        """
        Текстовое представление текущего вектора элементов
        """
        return str(self._vec)




class BaseOptimizer(object):
    """
    BaseOptimizer
    ---
    Класс базового оптимизатора
    """

    def __init__(self,
                 to_model_vec_size    : int,
                 from_model_vec_size  : int,
                 iter_limit           : int,
                 seed                 : int = None
                 ) -> None:
        """
        __init__
        ---
        Базовый конструктор класса оптимизатора
        
        Аргументы:
            to_model_vec_size    : int - Размерность вектора принимаемого целевой моделью оптимизации
            from_model_vec_size  : int - Размерность вектора получаемого от целевой модели оптимизации
            iter_limit           : int - Ограничение по количеству доступных итераций работы алгоритма
        """

        # Параметры генератора, доступные для перенастройки
        # ==========================================================================================

        self._to_model_vec_size    : int = to_model_vec_size
        """Размер входного вектора параметров"""
        self._from_model_vec_size  : int = from_model_vec_size + 1
        """Размер выходного вектора параметров"""
        self._iteration_limitation : int = iter_limit
        """Ограничение по количеству итераций алгоритма"""
        self._main_value_index     : int = 0
        """Индекс целевого оптимизируемого параметра"""
        self._seed                 : int = time.time() if seed is None else seed
        """Используемая база генератора для псевдослучайных последовательностей"""

        self._init_param()

        # Внутренние общие параметры генератора
        # ==========================================================================================
        self._historical_data_container : list = []
        """Лист хранения исторических данных выполнения алгоритмов оптимизации"""



    def __setattr__(self, key, value):
        """
        __setattr__
        ---
        Проверка корректности внесения атрибутов
        """
        # TODO: Добавить проверки атрибутов
        if key == "_to_model_vec_size":
            pass
        if key == "_from_model_vec_size":
            super().__setattr__(key, value+1)
        super().__setattr__(key, value)



    def __getattribute__(self, name):
        """
        __getattribute__
        ---
        Обработка доступа к атрибутам
        """
        
        # Доступ к методу _main_calc_func прописано добавление истории при каждом вызове
        if name == "_main_calc_func":
            self._collectIterHistoryData()
        
        return super().__getattribute__(name)



    def _init_param(self) -> None:
        """
        _init_param
        ---
        Функция инициализации параметров и массивов. 
        NOTE: Вынесено отдельно для упрощения переопределения функционала без необходимости 
              изменения коструктора базового класса
        """
        self._vec_candidates_size : int = 1
        """Число векторов кандидатов для получения решения. По умолчанию 1. Изменяется в зависимости
        от реализации."""

        self._to_opt_model_data : OptimizedVectorData = OptimizedVectorData(
            vec_size            = self._to_model_vec_size,
            vec_candidates_size = self._vec_candidates_size,
            seed                = self._seed
            )
        """Выходной вектор, отправляемый в модель оптимизации"""
        
        self._from_model_data : OptimizedVectorData = OptimizedVectorData(
            vec_size            = self._from_model_vec_size,
            vec_candidates_size = self._vec_candidates_size,
            seed                = self._seed
            )
        """Входной вектор, получаемый от модели оптимизации"""

        self._init_to_model_vec()



    def _init_to_model_vec(self) -> None:
        """
        _init_to_model_vec
        ---
        Внутренний метод инициализации выходного вектора _to_opt_model_vec
        
        Выполняет наполнение выходного массива.
        """
        self._to_opt_model_data.setVectorsRandVal(0.0, 1.0)



    def configure(self, **kwargs) -> None:
        """
        configure
        ---
        Метод настройки параметров работы оптимизатора
        
        Пример использования:
        
        >>> import SpecificOptimizer
        ... optimizer = SpecificOptimizer()
        ... optimizer.configure(some_parameter_A = some_value_A, some_parameter_B = some_value_B)
        """
        for key, value in kwargs.items():
            
            # Проверка наличия атрибута настройки параметров работы оптимизатора
            if key not in self.__dict__:
                raise KeyError(f"{self.__class__.__name__} не содержит настраиваемого параметра {key}")
            else:
                self.__dict__[key] = value



    def _main_calc_func(self) -> None:
        """
        _main_calc_func
        ---
        Главная функция выполнения расчетов алгоритма.
        
        Функция выполняет только одну итерацию расчетов.
        """
        raise NotImplementedError



    def _calc_objective_function_value(self) -> None:
        """
        _calc_objective_function_value
        ---
        Метод расчета значений целевой функции
        """
        raise NotImplementedError



    def modelOptimize(self, func : Callable[[np.array], np.array]) -> None:
        """
        modelOptimize
        ---
        Запуск оптимизации через передачу функции черного ящика
        
        TODO: Выполнить проверку пригодности функции, соразмерности возвращаемых векторов или 
              добавить автоопределение размености векторов
        """
        for _ in range(self._iteration_limitation):
            for to_vec, from_vec in zip(
                self._to_opt_model_data.iterVectors(), 
                self._from_model_data.iterVectors()
                ):
                from_vec[:] = func(to_vec)
            self._main_calc_func()



    def _collectIterHistoryData(self) -> None:
        """
        _collectIterHistoryData
        ---
        
        Внутренний метод сохранения информации о текущей итерации
        """
        loc_dict_item : dict = {}
        loc_dict_item["vec_to_model"]   = np.copy(self._to_opt_model_data.vecs)
        loc_dict_item["vec_from_model"] = np.copy(self._from_model_data.vecs)
        loc_dict_item["obj_val"]   = np.copy(self._from_model_data.vecs[self._main_value_index, :])
        self._historical_data_container.append(loc_dict_item)



    def getHistoricalData(self, key : None | Literal["vec_to_model", "vec_from_model", "obj_val"] = None) -> None | list:
        """
        getHistoricalData
        ---
        Получение исторических данных, собранных в результате работы алгоритма
        """
        if key is None:
            return self._historical_data_container
        
        if key not in ["vec_to_model", "vec_from_model", "obj_val"]:
            return None
        
        loc_output_list : list = []
        
        for item in self._historical_data_container:
            loc_output_list.append(item[key])
        return loc_output_list



    def getResult(self) -> np.ndarray:
        """
        getResult
        ---
        Метод получения результата работы выбранного алгоритма
        
        Выполняет возврат последнего сохраненного заначения, полученного путем применения алгоритма
        оптимизации.
        """
        return self._to_opt_model_data.vecs



    def setVecItemLimit(self, 
                        index : int, 
                        vec_dir : Literal["to_model", "from_model"] = "to_model",
                        min : None | float = None,
                        max : None | float = None) -> None:
        """
        setVecItemLimit
        ---
        
        Установка ограничения для параметра внутри вектора
        """
        if vec_dir == "to_model":
            self._to_opt_model_data.setLimitation(index = index, min = min, max = max)
        elif vec_dir == "from_model":
            self._from_model_data.setLimitation(index = index, min = min, max = max)
        else:
            ...


    def setVecItemType(
        self,
        index : int, 
        new_type : Literal["float", "bool"], 
        vec_dir : Literal["to_model", "from_model"] = "to_model",
        *args, 
        **kwargs
     )  :
        """
        setVecItemType
        ---
        
        Установка ограничения для параметра внутри вектора
        """
        if vec_dir == "to_model":
            self._to_opt_model_data.setVecItemType(index = index, *args, **kwargs)
        elif vec_dir == "from_model":
            self._from_model_data.setVecItemType(index = index, *args, **kwargs)
        else:
            ...



# Отладка функционала базового генератора
if __name__ == "__main__":
    # test_floatItem = _floatItem(min = 1.23, max = 56.3, seed = 123)
    # print(test_floatItem)
    # print(test_floatItem._isWitihRange(154))
    # print(test_floatItem._isWitihRange(4))
    # print(test_floatItem.isCorrect(4))
    # print(test_floatItem.isCorrect(4.0))
    # print(test_floatItem.isCorrect(123))
    # print(test_floatItem.isCorrect("Э"))

    test_OptimizedVectorData = OptimizedVectorData(vec_size = 12, vec_candidates_size = 5, seed = time.time())
    print("Initial")
    test_OptimizedVectorData.DEBUG_printInfo()
    test_OptimizedVectorData.setVecItemType(new_type="bool", index = 3)
    test_OptimizedVectorData.setVecItemType(new_type="bool", index = 8)
    test_OptimizedVectorData.setVecItemType(new_type="bool", index = 9)
    test_OptimizedVectorData.setLimitation(index= 3, min = 45, max = 75)
    test_OptimizedVectorData.setLimitation(index= 4, min = 45, max = 75)
    # test_OptimizedVectorData.setLimitation(index= 5, min = 485, max = 75)
    test_OptimizedVectorData.DEBUG_printInfo()
    test_OptimizedVectorData.setVectorRandValByLimits()
    test_OptimizedVectorData.DEBUG_printInfo()
    
    print(test_OptimizedVectorData.getInLimitsMatrix())
    test_OptimizedVectorData._vec[3, 3] = 8
    test_OptimizedVectorData.DEBUG_printInfo()
    print(test_OptimizedVectorData.getInLimitsMatrix())
    pass
