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

# Подключаемые модули зависимой конфигурации
if __name__ == "__main__":
    pass
else:
    pass


DEBUG_THE_BIGGEST_ONE = 999999999
"""Самое большое число. Пока в дебаге"""



class OptimizedVectorData(object):
    
    # Перечисление индексов во внутреннем массиве класса
    min_index          : int = 0
    max_index          : int = 1
    values_index_start : int = 2
    
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
        self._vec[:, OptimizedVectorData.min_index] = -np.inf
        self._vec[:, OptimizedVectorData.max_index] =  np.inf
        """Хранимый вектор значений, инициализируемый нулями, хранит значение минимума и максимума"""

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



    def setLimitation(self, 
                      index : int, 
                      min : None | float = None,
                      max : None | float = None) -> None:
        """
        setLimitation
        ---
        
        Установка ограничения для параметра внутри вектора
        """
        loc_min = min if min is not None else \
            -np.inf if self._vec[index][OptimizedVectorData.min_index] == -np.inf else \
                self._vec[index][OptimizedVectorData.min_index]
        loc_max = max if max is not None else \
            np.inf if self._vec[index][OptimizedVectorData.max_index] == np.inf else \
                self._vec[index][OptimizedVectorData.max_index]
        
        if loc_min is not None and loc_max is not None and loc_min >= loc_max:
            raise ValueError(f"Значение {loc_min} не может быть больше {loc_max}")
        
        self._vec[index][OptimizedVectorData.min_index] = loc_min
        self._vec[index][OptimizedVectorData.max_index] = loc_max



    def setVectorsRandVal(self, min_val : float, max_val : float) -> None:
        """
        setVectorsRandVal
        ---
        Получение начальных векторов с величинами по нормальному распределению
        """
        np.random.seed(int(self._seed))
        vec : np.array
        for vec in self.iterVectors():
            vec[:] = np.random.uniform(
                low  = min_val, 
                high = max_val, 
                size = self._vec_size
                ).copy()



    def setVectorRandValByLimits(self) -> None:
        """
        setVectorRandVal
        ---
        TODO: Получение начального вектора с величинами в диапазонах допустимого минимума и максимума 
        по нормальному распределению 
        """
        np.random.seed(int(self._seed))
        
        for vec, min, max in \
            zip(
                self._vec[:, OptimizedVectorData.values_index_start:],
                self._vec[:, OptimizedVectorData.min_index],
                self._vec[:, OptimizedVectorData.max_index],
                ):
            vec[:] = np.random.uniform(
                low  = min if min != -np.inf else -DEBUG_THE_BIGGEST_ONE, 
                high = max if max !=  np.inf else  DEBUG_THE_BIGGEST_ONE, 
                size = np.size(vec)
                ).copy()




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

        for vec, bool_vec in zip(self._vec, loc_matrix):
            for vec_item, bool_item_num in zip(vec[OptimizedVectorData.values_index_start:], range(np.size(bool_vec))):
                bool_vec[bool_item_num] = \
                    (vec_item > vec[OptimizedVectorData.min_index] and vec_item < vec[OptimizedVectorData.max_index])
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






# Отладка функционала базового генератора
if __name__ == "__main__":
    test_OptimizedVectorData = OptimizedVectorData(vec_size = 12, vec_candidates_size = 3, seed = time.time())
    # test_OptimizedVectorData.setVectorsRandVal(0.0, 1.0)
    test_OptimizedVectorData.setLimitation(4, 0.5, 1)
    test_OptimizedVectorData.setLimitation(5, 0.5, 1)
    test_OptimizedVectorData.setLimitation(6, 0.5, 1)
    test_OptimizedVectorData.setVectorRandValByLimits()
    for item in test_OptimizedVectorData.iterVectors():
        print(item)
        
    print(test_OptimizedVectorData.getInLimitsMatrix())
    
    
    
    # test_BaseOptimizer = BaseOptimizer(
    #     to_model_vec_size    = 5,
    #     from_model_vec_size  = 4,
    #     iter_limit           = 100,
    # )
    # print(test_BaseOptimizer._to_opt_model_data)
    # print(test_BaseOptimizer.getResult())
    
    
    # loc_vec = np.array([[1, 2, 3, 4]], dtype = float)
    # print(loc_vec)
    
    # loc_vec = np.append(loc_vec, [8, 9, 10])
    # loc_vec = np.extend(loc_vec, [[8, 9, 10, 114]], axis = 1)
    # print(loc_vec)
    
    
    # test_OptimizedVectorData = OptimizedVectorData(vec_size = 12, vec_candidates_size = 3)
    # print(test_OptimizedVectorData)

    # for item in test_OptimizedVectorData.iterVectors():
    #     print(item)
        
    # test_OptimizedVectorData.setVectorsRandVal(0.0, 1.0)
    # print(test_OptimizedVectorData)

    # for item in test_OptimizedVectorData.iterVectors():
    #     print(item)

    # test_BaseOptimizer = BaseOptimizer(
    #     to_model_vec_size    = 5,
    #     from_model_vec_size  = 4,
    #     iter_limit           = 100,
    # )
    # test_BaseOptimizer.modelOptimize(func = lambda: print(""))
    # print(test_BaseOptimizer._to_opt_model_data.vec)
    # print(test_BaseOptimizer._to_opt_model_data)
    # print(test_BaseOptimizer.vecToModel)
    # test_BaseOptimizer.vecToModel = 0
    # print(test_BaseOptimizer.vecFromModel)
    # test_BaseOptimizer.vecFromModel = np.array(np.zeros(76), dtype=float)
    pass
