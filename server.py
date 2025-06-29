from concurrent import futures
from typing import Dict, List, Any, Optional
import threading
import time
import numpy as np
import grpc
import uuid
import json
import os
import sys
from datetime import datetime

# Import necessary components from your RTO API
import RtoApi_pb2
import RtoApi_pb2_grpc
from optimization_methods import OPTIMIZATION_METHODS, SERVER_HOST, SERVER_PORT
from BlackBoxOptimizer import Optimizer, OptimisationTypes

class OptimizationSession:
    def __init__(self, session_id: str, session_data: Dict[str, Any]):
        self.session_id = session_id
        self.session_data = session_data
        self.status = "idle"
        self.current_iteration = 0
        self.start_time = 0
        self.thread = None
        self.stop_event = threading.Event()
        self.optimizer = None
        self.current_cv = None
        self.last_mv = None
        self.cv_ready = threading.Event()
        self.optimization_started = False
        self.optimization_finished = False
        self.session_file = f"session_{session_id}.json"
        self._init_optimizer()
        self.init_session_file()
        
    def init_session_file(self):
        """Инициализирует файл сессии"""
        with open(self.session_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "start_time": datetime.now().isoformat(),
                "optimization_method": self.session_data.get("optimization_method", "Unknown"),
                "MVs": self.session_data.get("MVs", []),
                "CVs": self.session_data.get("CVs", []),
                "iterations": []
            }, f, indent=2)

    def _init_optimizer(self):
        """Инициализация оптимизатора на основе выбранного метода"""
        method_name = self.session_data.get("optimization_method", "TestStepOpt")
        print(f"[SERVER] Выбран метод оптимизации: {method_name}")

        method_cfg = OPTIMIZATION_METHODS.get(method_name)
        if not method_cfg:
            raise ValueError(f"Метод оптимизации {method_name} не определен в настройках")

        opt_cls = method_cfg["class"]
        default_params = method_cfg.get("default_params", {})
        print(f"[SERVER] Параметры для метода: {default_params}")

        # Определяем тип оптимизации (минимизация или максимизация)
        is_maximize = self.session_data.get("maximize", False)
        opt_type = OptimisationTypes.maximize if is_maximize else OptimisationTypes.minimize
        print(f"[SERVER] Режим оптимизации: {'максимизация' if is_maximize else 'минимизация'}")

        # Получаем дискретные индексы (для булевых параметров)
        discrete_indices = []
        for idx, mv in enumerate(self.session_data["MVs"]):
            if mv.get("DataType") == "Boolean":
                discrete_indices.append(idx)
        
        if discrete_indices:
            print(f"[SERVER] Обнаружены дискретные параметры: {discrete_indices}")

        # Базовые параметры для всех методов
        base_params = {
            "seed": 1546,
            "to_model_vec_size": len(self.session_data["MVs"]),
            "from_model_vec_size": len(self.session_data["CVs"]),
            "iter_limit": self.session_data["max_iterations"],
        }

        # Дополнительные параметры для Optimizer
        optimizer_params = {
            "external_model": self.external_model,  # Передаем функцию-посредник
            "optimisation_type": opt_type,
            "target": None,
        }

        # Удаляем параметры, которые не поддерживаются классом GaussOpt
        method_specific_params = dict(default_params)
        if method_name == "GaussOpt":
            print(f"[SERVER] GaussOpt не поддерживает discrete_indices, этот параметр будет передан через configure")
        else:
            # Для остальных методов добавляем discrete_indices к параметрам
            method_specific_params["discrete_indices"] = discrete_indices

        # Создаем оптимизатор
        self.optimizer = Optimizer(
            optCls=opt_cls,
            **base_params,
            **optimizer_params,
            **method_specific_params
        )
        
        # Если это GaussOpt и есть дискретные индексы, настраиваем их через configure
        if method_name == "GaussOpt" and discrete_indices:
            self.optimizer.configure(discrete_indices=discrete_indices)

        # Устанавливаем границы для MV (входных переменных)
        for idx, mv in enumerate(self.session_data["MVs"]):
            lb = mv.get("LowerBound", -np.inf)
            ub = mv.get("UpperBound", np.inf)
            self.optimizer.setVecItemLimit(
                idx, "to_model", 
                min=lb,
                max=ub
            )
            print(f"[SERVER] MV[{idx}] '{mv.get('Name')}': min={lb}, max={ub}")
            
            # Установка дискретных параметров, если они есть
            if mv.get("DataType") == "Boolean":
                self.optimizer.setVecItemType(idx, "bool", "to_model")
                print(f"[SERVER] MV[{idx}] '{mv.get('Name')}' установлен как булевый тип")

        # Устанавливаем границы для CV (выходных переменных)
        print("[SERVER] Устанавливаем ограничения для CV в from_model со смещением (+1)")
        
        # ВАЖНОЕ ИСПРАВЛЕНИЕ: Смещаем индексы для CV для учета целевой функции на индексе 0
        # Первый CV (целевая функция) останется без ограничений, остальные CV смещаются на +1
        
        # Сначала установим "пустое" ограничение для индекса 0 (целевая функция)
        self.optimizer.setVecItemLimit(
            0, "from_model", 
            min=-np.inf,
            max=np.inf
        )
        print(f"[SERVER] CV[0] 'Target Function': min=-inf, max=inf (целевая функция)")
        
        # Затем устанавливаем ограничения для остальных CV с учетом смещения
        for idx, cv in enumerate(self.session_data["CVs"]):
            lb = cv.get("LowerBound", -np.inf)
            ub = cv.get("UpperBound", np.inf)
            # Смещение индекса: CV[i] будет на позиции i+1 в выходном векторе
            self.optimizer.setVecItemLimit(
                idx+1, "from_model", 
                min=lb,
                max=ub
            )
            print(f"[SERVER] CV[{idx}] '{cv.get('Name')}': индекс смещен на {idx+1}, min={lb}, max={ub}")

    def external_model(self, mv):
        """
        Функция-посредник между оптимизатором и внешней моделью (клиентом)
        Передает MV клиенту и получает CV обратно
        """
        # Если оптимизация завершена, возвращаем "плохое" значение
        if self.optimization_finished or self.stop_event.is_set():
            print("[SERVER][external_model] Optimization finished or stopped, returning large value")
            # Возвращаем массив с первым значением, отражающим "плохое" значение целевой функции
            return np.array([float('inf')] * len(self.session_data["CVs"]))
        
        # Сохраняем последний запрошенный MV для передачи клиенту
        self.last_mv = mv.copy()
        self.cv_ready.clear()
        
        print(f"[SERVER][external_model] Запрос оптимизатора на оценку MV: {mv}")
        
        # Ожидаем получения CV от клиента
        if not self.cv_ready.wait(timeout=30):
            print("[SERVER] Timeout waiting for CV values")
            return np.array([float('inf')] * len(self.session_data["CVs"]))
        
        # Проверка попадания в ограничения (только для логирования)
        within_bounds = True
        violations = []
        violation_count = 0
        total_violations = 0.0
        
        for i, cv_val in enumerate(self.current_cv):
            if i == 0:  # Пропускаем целевую функцию
                continue
                
            cv_info = self.session_data["CVs"][i-1]  # Корректируем индекс, поскольку у нас CV индексы смещены на 1
            lb = cv_info.get("LowerBound", -np.inf)
            ub = cv_info.get("UpperBound", np.inf)
            
            if cv_val < lb:
                within_bounds = False
                violation_val = lb - cv_val
                violations.append(f"CV[{i}] '{cv_info.get('Name')}': {cv_val:.4f} < {lb:.4f} (нарушение: {violation_val:.4f})")
                violation_count += 1
                total_violations += violation_val
            elif cv_val > ub:
                within_bounds = False
                violation_val = cv_val - ub
                violations.append(f"CV[{i}] '{cv_info.get('Name')}': {cv_val:.4f} > {ub:.4f} (нарушение: {violation_val:.4f})")
                violation_count += 1
                total_violations += violation_val

        if within_bounds:
            print(f"[SERVER][external_model] Полученные CV в пределах ограничений. Целевая функция: {self.current_cv[0]:.6f}")
            
            # Сохраняем лучший допустимый MV и значение целевой функции
            if not hasattr(self, 'best_feasible_mv') or not hasattr(self, 'best_feasible_target'):
                self.best_feasible_mv = self.last_mv.copy()
                self.best_feasible_target = self.current_cv[0]
            elif self.session_data.get("maximize", False) and self.current_cv[0] > self.best_feasible_target:
                # Для задачи максимизации
                self.best_feasible_mv = self.last_mv.copy()
                self.best_feasible_target = self.current_cv[0]
            elif not self.session_data.get("maximize", False) and self.current_cv[0] < self.best_feasible_target:
                # Для задачи минимизации
                self.best_feasible_mv = self.last_mv.copy()
                self.best_feasible_target = self.current_cv[0]
        else:
            print(f"[SERVER][external_model] ВНИМАНИЕ: CV вне пределов ограничений! Нарушено {violation_count} ограничений (сумм. величина: {total_violations:.4f})")
            for v in violations:
                print(f"  - {v}")

        # Возвращаем значения CV без модификаций - обработка штрафов выполняется в методах оптимизации
        return np.array(self.current_cv)
        
    def _run_optimization(self):
        """Запуск оптимизации в отдельном потоке"""
        try:
            self.start_time = time.time()
            self.status = "running"
            
            # Запуск оптимизации - метод сам обрабатывает итерации
            self.optimizer.modelOptimize()
            
            self.optimization_finished = True
            self.status = "completed"
            
            print(f"[SERVER] Optimization completed in {time.time() - self.start_time:.2f} seconds")
            
        except Exception as e:
            self.status = "error"
            print(f"[SERVER] Error during optimization: {e}")
            import traceback
            traceback.print_exc()

    def process_iteration(self, cv_value=None):
        """
        Обработка итерации оптимизации
        Если cv_value не None, значит клиент отправляет результат оценки
        Иначе - клиент запрашивает новые значения MV
        """
        # Когда клиент предоставляет значения CV от оценки MV
        if cv_value is not None:
            self.current_cv = cv_value
            self.cv_ready.set()
            self.current_iteration += 1

            # Отслеживание идентичных итераций для обнаружения зацикливания
            if not hasattr(self, 'last_iterations'):
                self.last_iterations = []
                self.repeat_count = 0

            if self.last_mv is not None:
                mv_tuple = tuple(np.round(self.last_mv, 6))
                if len(self.last_iterations) > 0 and mv_tuple == self.last_iterations[-1]:
                    self.repeat_count += 1
                    print(f"[SERVER] ВНИМАНИЕ: Зацикливание обнаружено (итерация {self.repeat_count})")
                    if self.repeat_count >= 5:
                        print("[SERVER] Принудительное изменение MV для выхода из цикла")
                        if self.session_data["optimization_method"] == "GaussOpt":
                            perturbation = np.random.uniform(-0.05, 0.05, len(self.last_mv))
                            perturbed_mv = self.last_mv + perturbation * np.abs(self.last_mv)
                            for i, mv in enumerate(self.session_data["MVs"]):
                                lb = mv.get("LowerBound", -np.inf)
                                ub = mv.get("UpperBound", np.inf)
                                perturbed_mv[i] = np.clip(perturbed_mv[i], lb, ub)
                            self.last_mv = perturbed_mv
                            print(f"[SERVER] Возмущенные MV: {self.last_mv}")
                            self.repeat_count = 0
                            self.cv_ready.clear()
                else:
                    self.repeat_count = 0
                    self.last_iterations.append(mv_tuple)
                    if len(self.last_iterations) > 3:
                        self.last_iterations.pop(0)

            # Сохраняем данные итерации
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            session_data["iterations"].append({
                "iteration": self.current_iteration,
                "mv_values": self.last_mv.tolist() if isinstance(self.last_mv, np.ndarray) else self.last_mv,
                "cv_values": self.current_cv,
                "timestamp": datetime.now().isoformat()
            })
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)

        # Проверка на завершение
        if self.optimization_finished or self.current_iteration >= self.session_data["max_iterations"]:
            self.optimization_finished = True
            try:
                result = None
                # Для GaussOpt всегда используем только getResult()
                if self.session_data["optimization_method"] == "GaussOpt":
                    result = self.optimizer.getResult()
                    print(f"[SERVER] GaussOpt: получен результат через getResult()")
                    if result is None or (hasattr(result, "__len__") and len(result) == 0):
                        print(f"[SERVER] GaussOpt: getResult() вернул пустой результат, используем последний MV")
                        result = self.last_mv
                # Для остальных алгоритмов логика прежняя
                elif hasattr(self, 'best_feasible_mv') and self.best_feasible_mv is not None:
                    print(f"[SERVER] Используем лучшее найденное допустимое решение")
                    result = self.best_feasible_mv
                else:
                    result = self.optimizer.getResult()
                    if result is None or (hasattr(result, "__len__") and len(result) == 0):
                        print(f"[SERVER] getResult() вернул пустой результат, используем последний MV")
                        result = self.last_mv
            except Exception as e:
                print(f"[SERVER] Ошибка при получении результата оптимизации: {e}")
                
                result = self.last_mv

            # Проверка границ результата
            try:
                if hasattr(result, '__iter__'):
                    for i, mv_val in enumerate(result):
                        if i < len(self.session_data["MVs"]):
                            lb = self.session_data["MVs"][i].get("LowerBound", -np.inf)
                            ub = self.session_data["MVs"][i].get("UpperBound", np.inf)
                            if mv_val < lb or mv_val > ub:
                                print(f"[SERVER] ВНИМАНИЕ: MV[{i}]={mv_val} вне границ [{lb}, {ub}]")
            except Exception as e:
                print(f"[SERVER] Ошибка при проверке границ результата: {e}")

            print(f"[SERVER] Optimization completed. Best MV: {result}")
            return result.tolist() if hasattr(result, "tolist") else list(result)

        # Первая итерация - запуск оптимизации в отдельном потоке
        if not self.optimization_started:
            self.optimization_started = True
            self.thread = threading.Thread(target=self._run_optimization)
            self.thread.daemon = True
            self.thread.start()
            timeout = time.time() + 10
            while self.last_mv is None and time.time() < timeout:
                time.sleep(0.01)
            if self.last_mv is None:
                raise TimeoutError("Оптимизатор не запросил первые значения MV вовремя")
            print(f"[SERVER] Начальные MV от оптимизатора: {self.last_mv}")
            return self.last_mv.tolist()

        timeout = time.time() + 1
        wait_count = 0
        while self.cv_ready.is_set() and time.time() < timeout:
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 50:
                print("[SERVER] Предупреждение: Длительное ожидание потребления CV")
        if self.cv_ready.is_set():
            print("[SERVER] Предупреждение: Оптимизатор не использовал значение CV")
            if self.session_data["optimization_method"] == "GaussOpt" and hasattr(self, 'repeat_count') and self.repeat_count > 0:
                print("[SERVER] Принудительный сброс CV_ready для GaussOpt")
                self.cv_ready.clear()
        print(f"[SERVER] Следующий MV для клиента: {self.last_mv}")
        return self.last_mv.tolist()

class RtoService(RtoApi_pb2_grpc.RtoServiceServicer):
    def __init__(self):
        """Инициализация сервиса RTO"""
        self.sessions = {}  # Словарь для хранения сессий оптимизации
        self.service_start_time = datetime.now()
        print(f"[SERVER] RTO Service initialized at {self.service_start_time}")

    def GetServiceStatus(self, request, context):
        """Возвращает статус сервиса"""
        try:
            uptime_seconds = (datetime.now() - self.service_start_time).total_seconds()
            response = RtoApi_pb2.ServiceStatus(
                serviceId="RTO-Optimization-Service",
                status="active",
                serviceType="Optimization",
                user="system",
                message=f"Active sessions: {len(self.sessions)}"
            )
            return response
        except Exception as e:
            print(f"[SERVER] Error getting service status: {e}")
            return RtoApi_pb2.ServiceStatus(
                serviceId="RTO-Optimization-Service",
                status="error",
                serviceType="Optimization",
                user="system",
                message=f"Error: {str(e)}"
            )

    def StartOptimizeSession(self, request, context):
        """Запуск новой сессии оптимизации"""
        try:
            # Создаем новую сессию с уникальным ID
            session_id = str(uuid.uuid4())
            
            # Подготавливаем данные сессии из запроса
            session_data = {
                "MVs": [],
                "CVs": [],
                "max_iterations": request.max_iterations,
                "optimization_method": request.optimization_method,
                "maximize": request.maximize
            }
            
            # Добавляем информацию о MV (управляемых переменных)
            for mv in request.mvs:
                session_data["MVs"].append({
                    "Id": mv.id,
                    "Name": mv.name,
                    "DataType": mv.dataType,  # Обратите внимание: dataType, а не data_type
                    "LowerBound": mv.lower_bound,
                    "UpperBound": mv.upper_bound,
                    "InitialValue": 0.0  # В proto нет initial_value, устанавливаем по умолчанию
                })
            
            # Добавляем информацию о CV (контролируемых переменных)
            for cv in request.cvs:
                session_data["CVs"].append({
                    "Id": cv.id,
                    "Name": cv.name,
                    "DataType": cv.dataType,  # dataType, а не data_type
                    "LowerBound": cv.lower_bound,
                    "UpperBound": cv.upper_bound
                })
            
            # Создаем новую сессию оптимизации
            self.sessions[session_id] = OptimizationSession(session_id, session_data)
            
            print(f"[SERVER] Started new optimization session: {session_id}")
            print(f"[SERVER] Method: {request.optimization_method}, MVs: {len(session_data['MVs'])}, CVs: {len(session_data['CVs'])}")
            
            # Создаем ответ в соответствии с proto
            response = RtoApi_pb2.StartResponse(
                optimization_instance_id=session_id,
                is_good=True,
                message="Session started successfully"
            )
            return response
                
        except Exception as e:
            print(f"[SERVER] Error starting optimization session: {e}")
            import traceback
            traceback.print_exc()
            response = RtoApi_pb2.StartResponse(
                optimization_instance_id="",
                is_good=False,
                message=f"Failed to start session: {str(e)}"
            )
            return response

    def OptimizeIteration(self, request, context):
        """Обработка одной итерации оптимизации"""
        try:
            session_id = request.optimization_instance_id
            if session_id not in self.sessions:
                return RtoApi_pb2.OptimizeIterationResponse(
                    flag=4,  # Ошибка
                    message=f"Session {session_id} not found",
                    sessionId=session_id
                )
            
            session = self.sessions[session_id]
            
            # Если клиент отправляет результаты оценки
            cv_values = None
            if len(request.cv_values) > 0:
                # Правильное объединение CV: сначала целевая функция, затем ограничения
                if request.HasField("objective_function_value"):
                    obj_val = request.objective_function_value.numericValue
                    cv_values = [obj_val] + [cv.numericValue for cv in request.cv_values]
                    print(f"[SERVER] Получены значения CV от клиента: объективная={obj_val}, ограничения={[cv.numericValue for cv in request.cv_values]}")
                else:
                    cv_values = [cv.numericValue for cv in request.cv_values]
                    print(f"[SERVER] Получены только значения CV-ограничений от клиента: {cv_values}")
        
            # Обработка итерации и получение новых MV
            mv_values = session.process_iteration(cv_values)
            
            # Определяем статус оптимизации
            flag = 0  # Успешная итерация
            if session.optimization_finished:
                flag = 3  # Оптимизация завершена
        
            # Формирование ответа
            response = RtoApi_pb2.OptimizeIterationResponse(
                flag=flag,
                message="Iteration processed" if not session.optimization_finished else "Optimization completed",
                sessionId=session_id
            )
            
            # Добавление значений MV в ответ
            for i, mv_val in enumerate(mv_values):
                mv = RtoApi_pb2.TagVal(
                    tagId=session.session_data["MVs"][i]["Id"], 
                    numericValue=float(mv_val),
                    isGood=True
                )
                response.mv_values.append(mv)
            
            return response
        
        except Exception as e:
            print(f"[SERVER] Error during optimization iteration: {e}")
            import traceback
            traceback.print_exc()
            return RtoApi_pb2.OptimizeIterationResponse(
                flag=-1,  # Ошибка
                message=f"Failed to process iteration: {str(e)}",
                sessionId=request.optimization_instance_id if hasattr(request, "optimization_instance_id") else ""
            )

    def Pause(self, request, context):
        """Приостановка сессии оптимизации"""
        try:
            session_id = request.sessionId
            if session_id not in self.sessions:
                return RtoApi_pb2.PauseResponse(
                    message=f"Session {session_id} not found",
                    is_paused=False
                )
            
            session = self.sessions[session_id]
            session.status = "paused"
            
            return RtoApi_pb2.PauseResponse(
                message=f"Session {session_id} paused",
                is_paused=True
            )
            
        except Exception as e:
            return RtoApi_pb2.PauseResponse(
                message=f"Failed to pause session: {str(e)}",
                is_paused=False
            )

    def Stop(self, request, context):
        """Остановка сессии оптимизации"""
        try:
            session_id = request.sessionId
            if session_id not in self.sessions:
                return RtoApi_pb2.StopResponse(
                    message=f"Session {session_id} not found"
                )
            
            session = self.sessions[session_id]
            session.stop_event.set()
            session.status = "stopped"
            
            return RtoApi_pb2.StopResponse(
                message=f"Session {session_id} stopped"
            )
            
        except Exception as e:
            return RtoApi_pb2.StopResponse(
                message=f"Failed to stop session: {str(e)}"
            )

def serve():
    """Запуск gRPC сервера с настройкой через конфигурационный файл"""
    # Определяем путь к config файлу рядом с исполняемым файлом (работает и как скрипт, и как exe)
    if getattr(sys, 'frozen', False):
        # Если запущен как exe
        app_path = os.path.dirname(sys.executable)
    else:
        # Если запущен как скрипт
        app_path = os.path.dirname(os.path.abspath(__file__))
    
    # Путь к конфигурационному файлу
    config_path = os.path.join(app_path, 'server_config.json')
    
    # Значения по умолчанию
    port = SERVER_PORT
    
    # Пытаемся загрузить настройки из файла
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                if 'port' in config:
                    port = config['port']
                    print(f"[SERVER] Загружена настройка порта из файла конфигурации: {port}")
        except Exception as e:
            print(f"[SERVER] Ошибка загрузки конфигурации: {e}, используется порт по умолчанию: {port}")
    else:
        # Создаем файл конфигурации по умолчанию
        try:
            with open(config_path, 'w') as config_file:
                json.dump({'port': SERVER_PORT}, config_file, indent=2)
                print(f"[SERVER] Создан файл конфигурации по умолчанию: {config_path}")
        except Exception as e:
            print(f"[SERVER] Не удалось создать файл конфигурации: {e}, используется порт по умолчанию: {port}")
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    RtoApi_pb2_grpc.add_RtoServiceServicer_to_server(RtoService(), server)
    server_address = f'[::]:{port}'
    server.add_insecure_port(server_address)
    server.start()
    print(f"[SERVER] Сервер запущен и слушает порт {port}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()