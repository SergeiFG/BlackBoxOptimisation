import grpc
from concurrent import futures
import RtoApi_pb2
import RtoApi_pb2_grpc
import threading
import time
import os
import uuid
from typing import Dict
import re
from math import (
    sin, cos, tan, asin, acos, atan,
    exp, sqrt, log10, pow as math_pow
)
from datetime import datetime
import subprocess
import json
import numpy as np

class OptimizationSession:
    def __init__(self, session_id, session_data):
        self.session_id = session_id
        self.session_data = session_data
        self.status = "idle"
        self.current_iteration = 0
        self.start_time = 0
        self.thread = None
        self.stop_event = threading.Event()
        self.session_file = f"session_{session_id}.txt"
        self._init_session_file()
        self.current_mvs = None

    def _init_session_file(self):
        """Инициализирует файл сессии с информацией о методе"""
        with open(self.session_file, 'w') as f:
            f.write(f"=== Optimization Session {self.session_id} ===\n")
            f.write(f"Method: {self.session_data['optimization_method']}\n")
            f.write(f"Config: {json.dumps(self.session_data['method_config'], indent=2)}\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model ID: {self.session_data.get('model_id', 'N/A')}\n")
            f.write(f"Objective: {self.session_data['objective_function']}\n")
            f.write(f"Maximize: {self.session_data['maximize']}\n")
            f.write(f"Max iterations: {self.session_data['max_iterations']}\n\n")
            f.write("=== Session Log ===\n")


    def _log_event(self, message):
        """Логирует событие в файл и выводит в консоль"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}\n"
        print(f"Session {self.session_id}: {log_msg.strip()}")
        with open(self.session_file, 'a') as f:
            f.write(log_msg)

    def _run_external_optimization(self, initial_mvs=None):
        """Запускает внешний оптимизатор с учетом выбранного метода"""
        try:
            # Берем только первую MV (в запросе заказчика она одна)
            mv = self.session_data['MVs'][0]
            
            params = {
                'method': self.session_data['optimization_method'],
                'to_model_vec_size': 1,  # Только одна переменная
                'iter_limit': self.session_data['max_iterations'],
                'initial_values': [initial_mvs[0]] if initial_mvs else [0.0],
                'objective_function': self.session_data['objective_function'],
                'mv_ids': [mv.id],  # ID единственной MV
                'bounds': [[mv.lower_bound, mv.upper_bound]]  # Границы для одной переменной
            }
            
            process = subprocess.Popen(
                ['python', 'OptimizerWrapper.py'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            process.stdin.write(json.dumps(params))
            process.stdin.close()
            
            stdout, stderr = process.communicate()
            
            for line in stdout.split('\n'):
                if line.startswith('FINAL_RESULT:'):
                    result = json.loads(line[13:])
                    if result['status'] == 'success':
                        return result['optimized_values']
                elif line.startswith('ERROR:'):
                    error = json.loads(line[6:])
                    raise RuntimeError(error['error'])
            
            raise RuntimeError("No valid result found in optimizer output")
        except Exception as e:
            self._log_event(f"Optimizer error: {str(e)}")
            raise
    def optimize(self, initial_mvs=None):
        """Метод оптимизации с защитой от ошибок кодировки"""
        self.status = "running"
        self._log_event("OPTIMIZATION STARTED")
        
        try:
            # Получаем имена переменных
            mv_names = [mv.name for mv in self.session_data['MVs']]
            self._log_event(f"Optimizing variables: {', '.join(mv_names)}")
            
            # Запускаем оптимизацию
            optimized_values = self._run_external_optimization(
                initial_mvs or [0.5]*len(mv_names)
            )
            
            # Логируем результаты
            self._log_event("OPTIMIZATION COMPLETED SUCCESSFULLY")
            self._log_event("Final optimized values:")
            for name, value in zip(mv_names, optimized_values):
                if isinstance(value, (list, np.ndarray)):
                    value_str = ", ".join([f"{x:.6f}" for x in value])
                else:
                    value_str = f"{value:.6f}"
                self._log_event(f"  {name}: {value_str}")
            
        except Exception as e:
            self.status = "error"
            self._log_event(f"OPTIMIZATION FAILED: {str(e)}")
            return None
        
class DataProcessingService(RtoApi_pb2_grpc.DataProcessingServiceServicer):
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.available_methods = {
            "GradientDescent": {
                "module": "TestStepOpt",
                "params": {"step": 0.01}
            },
            "Evolutionary": {
                "module": "EvolutionaryOpt",
                "params": {
                    "population_size": 30,
                    "offspring_per_parent": 2,
                    "mutation_prob": 0.3,
                    "sigma_init": 0.2
                }
            }
        }
        print("Сервер RTO инициализирован. Доступные методы:", list(self.available_methods.keys()))

    def GetServiceStatus(self, request, context):
        return RtoApi_pb2.ServiceStatus(
            serviceId="",
            status="Running",
            serviceType="RtoService",
            user="",
            message=""
        )

    def Start(self, request, context):
        try:
            # Проверяем, что метод существует
            if request.optimization_method not in self.available_methods:
                raise ValueError(f"Unknown method: {request.optimization_method}. Available: {list(self.available_methods.keys())}")
            
            session_id = str(uuid.uuid4())
            session_data = {
                "CVs": request.cvs,
                "MVs": request.mvs,
                "objective_function": request.objective_function,
                "maximize": request.maximize,
                "optimization_method": request.optimization_method,
                "method_config": self.available_methods[request.optimization_method],
                "max_iterations": request.max_iterations,
                "model_id": request.model_id
            }
            
            with self.lock:
                self.sessions[session_id] = OptimizationSession(session_id, session_data)
                print(f"Created new session {session_id} with method {request.optimization_method}")

            return RtoApi_pb2.StartResponse(
                optimization_instance_id=session_id,
                is_good=True,
                message=f"Session created with method {request.optimization_method}"
            )
        except Exception as e:
            print(f"Error creating session: {str(e)}")
            return RtoApi_pb2.StartResponse(
                optimization_instance_id="",
                is_good=False,
                message=str(e)
            )

    def StartOptimize(self, request, context):
        try:
            session_id = request.optimization_instance_id
            if session_id not in self.sessions:
                return RtoApi_pb2.StartOptimizeResponse(
                    MVs=[],
                    sessionId="",
                    flag=-1,
                    message="err_sessionNotFound"
                )

            session = self.sessions[session_id]
            
            if session.status == "running":
                return RtoApi_pb2.StartOptimizeResponse(
                    MVs=[],
                    sessionId=session_id,
                    flag=2,
                    message="err_optimization_already_running"
                )
            elif session.status == "paused":
                session.status = "running"
                session._log_event("OPTIMIZATION RESUMED")
                return RtoApi_pb2.StartOptimizeResponse(
                    MVs=[],
                    sessionId=session_id,
                    flag=1,
                    message="msg_optimization_resumed"
                )

            # Получаем начальные значения MVs если они переданы
            initial_mvs = None
            if request.initial_mvs:
                initial_mvs = [mv.numericValue for mv in request.initial_mvs]
            
            # Запускаем оптимизацию в отдельном потоке
            session.stop_event.clear()
            session.thread = threading.Thread(
                target=session.optimize,
                kwargs={'initial_mvs': initial_mvs}
            )
            session.thread.start()

            return RtoApi_pb2.StartOptimizeResponse(
                MVs=[],
                sessionId=session_id,
                flag=0,
                message="msg_optimization_started"
            )
        except Exception as e:
            print(f"Optimization start error: {str(e)}")
            return RtoApi_pb2.StartOptimizeResponse(
                MVs=[],
                sessionId="",
                flag=-1,
                message=f"Err_{str(e)}"
            )

    def Pause(self, request, context):
        try:
            session_id = request.sessionId
            if session_id not in self.sessions:
                return RtoApi_pb2.PauseResponse(
                    message="err_sessionNotFound",
                    is_paused=False
                )

            session = self.sessions[session_id]
            
            if session.status == "running":
                session.status = "paused"
                session._log_event("OPTIMIZATION PAUSED")
                return RtoApi_pb2.PauseResponse(
                    message="msg_paused",
                    is_paused=True
                )
            elif session.status == "paused":
                return RtoApi_pb2.PauseResponse(
                    message="msg_already_paused",
                    is_paused=True
                )
            else:
                return RtoApi_pb2.PauseResponse(
                    message="err_optimization_not_running",
                    is_paused=False
                )
        except Exception as e:
            print(f"Pause error: {str(e)}")
            return RtoApi_pb2.PauseResponse(
                message=f"Err_{str(e)}",
                is_paused=False
            )

    def Stop(self, request, context):
        try:
            session_id = request.sessionId
            if session_id not in self.sessions:
                return RtoApi_pb2.StopResponse(
                    message="err_sessionNotFound"
                )

            session = self.sessions[session_id]
            session.stop_event.set()
            session._log_event("SESSION STOPPED")
            
            # Удаляем файл сессии
            try:
                if os.path.exists(session.session_file):
                    os.remove(session.session_file)
                    print(f"Файл сессии удален: {session.session_file}")
            except Exception as e:
                print(f"Ошибка при удалении файла сессии: {str(e)}")
            
            if session.thread and session.thread.is_alive():
                session.thread.join(timeout=1.0)
            
            with self.lock:
                if session_id in self.sessions:
                    del self.sessions[session_id]
                    print(f"Session stopped: {session_id}")

            return RtoApi_pb2.StopResponse(
                message="msg_session_stopped"
            )
        except Exception as e:
            print(f"Stop error: {str(e)}")
            return RtoApi_pb2.StopResponse(
                message=f"Err_{str(e)}"
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    RtoApi_pb2_grpc.add_DataProcessingServiceServicer_to_server(
        DataProcessingService(), server)
    server.add_insecure_port('[::]:5081')
    server.start()
    print("Сервер запущен и слушает порт 5081")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()