"""
BaseOptimizer
---

Файл базового оптимизатора

Определяет общие методы и атрибуты для всех частных оптимизаторов

"""

# Подключаемые модули зависимой конфигурации
if __name__ == "__main__":
    import numpy as np
else:
    import numpy as np



class OptimizedVectorData(object):
    
    # Перечисление индексов во внутреннем массиве класса
    min_index          : int = 0
    max_index          : int = 1
    values_index_start : int = 2
    
    # Полежение векторов
    axis_X : int = 0
    axis_Y : int = 1

    def __init__(
        self,
        vec_size : int,
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



    def iter_vectors(self) -> np.array:
        """
        iter_vectors
        ---
        Итератор входящих вектров, без лимитирующих векторов
        """
        # TODO: Подумать над управлением доступом
        for column in range( \
            np.size(self._vec[OptimizedVectorData.axis_Y]) - OptimizedVectorData.values_index_start):
            yield self._vec[:, column + OptimizedVectorData.values_index_start]



    @property
    def vec(self) -> np.array:
        """
        Возврат вектора одних только значений.
        
        Производит отсечение столбцов значений максимума и минимума доступного для параметра
        """
        return self._vec[:, OptimizedVectorData.values_index_start]


    @vec.setter
    def vec(self, vec) -> None:
        # TODO: Добавить проверки данных для присвоения
        self._vec[:, OptimizedVectorData.values_index_start] = vec


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


    def setVectorRandVal(self, min_val : float, max_val : float) -> None:
        """
        setVectorRandVal
        ---
        Получение начального вектора с величинами по нормальному распределению
        """
        self.vec =  np.random.uniform(
            low  = min_val, 
            high = max_val, 
            size = np.shape(self.vec)
            )


    def setVectorRandValByLimits(self) -> None:
        """
        setVectorRandVal
        ---
        TODO: Получение начального вектора с величинами в диапазонах допустимого минимума и максимума 
        по нормальному распределению 
        """
        pass


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
        self._from_model_vec_size  : int = from_model_vec_size
        """Размер выходного вектора параметров"""
        self._iteration_limitation : int = iter_limit
        """Ограничение по количеству итераций алгоритма"""

        # Внутренние общие параметры генератора
        # ==========================================================================================

        self._to_opt_model_data : OptimizedVectorData = OptimizedVectorData(
            size = self._to_model_vec_size
            )
        """Выходной вектор, отправляемый в модель оптимизации"""
        
        self._from_model_data : OptimizedVectorData = OptimizedVectorData(
            size = self._from_model_vec_size
            )
        """Входной вектор, получаемый от модели оптимизации"""
        
        self._objective_function_value_from_model : float = 0.0
        """Значение целевой функции"""
        
        self._init_param()



    def _init_param(self) -> None:
        """
        _init_param
        ---
        Функция инициализации параметров и массивов. 
        NOTE: Вынесено отдельно для упрощения переопределения функционала без необходимости 
              изменения коструктора базового класса
        """
        self._init_to_model_vec()



    def _init_to_model_vec(self) -> None:
        """
        _init_to_model_vec
        ---
        Внутренний метод инициализации выходного вектора _to_opt_model_vec
        
        Выполняет наполнение выходного массива.
        """
        self._to_opt_model_data.setVectorRandVal(0.0, 1.0)



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



    def AlgIter(self):
        """
        AlgIter
        ---
        Главный итератор решателя
        
        На каждой итерации вызова выполняется возврат указателей на приведенные векторы типа 
        np.array. Для изменений из вне доступны только вектора _from_opt_model_vec и 
        _objective_function_value
        
        Возврат:
            _to_model_vec            : np.array - Вектор для отправки в модель оптимизации
            _from_model_vec          : np.array - Вектор полученный от модели оптимзации
            _from_model_obj_func_val : np.array - Значение целевой функции от модели оптимизации
        """
        # TODO: Пересмотреть логику работы итератора, а сейчас запрет использования
        raise NotImplementedError
        
        for _ in range(self.iteration_limitation):
            yield (self._to_opt_model_vec.copy(), self._from_opt_model_vec, self._objective_function_value)
            self._main_calc_func()



    @property
    def vecToModel(self) -> np.array:
        """Предоставляемый вектор для отправки в модель"""
        self._main_calc_func()
        return self._to_opt_model_data.vec.copy()


    @property
    def vecFromModel(self) -> None:
        return self._from_model_data.vec.copy()
        # raise AttributeError("Чтение атрибута не допускается")


    @vecFromModel.setter
    def vecFromModel(self, new_value : np.array) -> None:
        """Установка значений, полученных от модели"""
        
        if new_value is None:
            return
        if not isinstance(new_value, np.ndarray):
            raise TypeError("Неврный тип параметра для присовения атрибуту vecFromModel")
        if self._from_model_data.vec.dtype != new_value.dtype:
            raise TypeError("Неверный тип элементов присваиваемого вектора")
        if self._from_model_data.vec.shape != new_value.shape:
            raise TypeError("Неверная размерность присваемого вектора")

        self._from_model_data.vec = new_value.copy()


    @property
    def objFuncValue(self) -> float | None:
        """
        Чтиение установленного значения целевой функции
        """
        return self._objective_function_value_from_model


    @objFuncValue.setter
    def objFuncValue(self, new_val : float | int | None) -> None:
        """
        Установка занчения целевой функции от модели
        """
        if new_val is None:
            return
        if not isinstance(new_val, float) and not isinstance(new_val, int):
            raise TypeError("Присваивается неверный тип атрибуту _objective_function_value_from_model")
        self._objective_function_value_from_model = float(new_val)


    def getResult(self):
        """
        getResult
        ---
        Метод получения результата работы выбранного алгоритма
        """
        raise NotImplementedError



# Отладка функционала базового генератора
if __name__ == "__main__":
    test_OptimizedVectorData = OptimizedVectorData(vec_size = 12, vec_candidates_size = 3)
    print(test_OptimizedVectorData)

    for item in test_OptimizedVectorData.iter_vectors():
        print(item)


    # test_BaseOptimizer = BaseOptimizer(
    #     to_model_vec_size    = 5,5
    #     from_model_vec_size  = 4,
    #     iter_limit           = 100,
    # )
    # print(test_BaseOptimizer._to_opt_model_data.vec)
    # print(test_BaseOptimizer._to_opt_model_data)
    # print(test_BaseOptimizer.vecToModel)
    # test_BaseOptimizer.vecToModel = 0
    # print(test_BaseOptimizer.vecFromModel)
    # test_BaseOptimizer.vecFromModel = np.array(np.zeros(76), dtype=float)
    pass
