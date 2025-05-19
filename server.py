import grpc
from concurrent import futures
import RtoApi_pb2
import RtoApi_pb2_grpc
import threading
import time
import os
import uuid
import numpy as np
from typing import Dict
import re
from math import (
    sin, cos, tan, asin, acos, atan,
    exp, sqrt, log10, pow as math_pow
)
from datetime import datetime
from optimization_methods import OPTIMIZATION_METHODS

class OptimizationSession:
    def __init__(self, session_id, session_data):
        self.session_id = session_id
        self.session_data = session_data
        self.status = "idle"
        self.current_iteration = 0
        self.start_time = 0
        self.thread = None
        self.stop_event = threading.Event()
        self.optimizer = None
        self.current_cv = None
        self.last_mv = None  # Добавлено: хранить последний MV, который ждёт расчёта
        self.cv_ready = threading.Event()  # Сигнал, что CV получено
        self._init_optimizer()
        
    def init_session_file(self):
        """Инициализирует файл сессии"""
        with open(self.session_file, 'w') as f:
            f.write(f"=== Optimization Session {self.session_id} ===\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model ID: {self.session_data.get('model_id', 'N/A')}\n")
            f.write(f"Maximize: {self.session_data['maximize']}\n")
            f.write(f"Method: {self.session_data['optimization_method']}\n")
            f.write(f"Max iterations: {self.session_data['max_iterations']}\n\n")
            f.write("=== Session Log ===\n")

    def _init_optimizer(self):
        from BlackBoxOptimizer import Optimizer, OptimisationTypes

        method_name = self.session_data.get("optimization_method", "TestStepOpt")
        print(f"[SERVER] Выбран метод оптимизации: {method_name}")  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ

        method_cfg = OPTIMIZATION_METHODS.get(method_name)
        if not method_cfg:
            raise ValueError(f"Unknown optimization method: {method_name}")

        opt_cls = method_cfg["class"]
        default_params = method_cfg.get("default_params", {})
        print(f"[SERVER] Параметры для метода: {default_params}")  # <-- И ЭТУ

        opt_type = OptimisationTypes.maximize if self.session_data.get("maximize", False) else OptimisationTypes.minimize

        self.optimizer = Optimizer(
            optCls=opt_cls,
            seed=1546,
            to_model_vec_size=len(self.session_data["MVs"]),
            from_model_vec_size=len(self.session_data["CVs"]),
            iter_limit=self.session_data["max_iterations"],
            external_model=self.external_model,
            optimisation_type=opt_type,
            target=None,
            **default_params
        )

        # После создания Optimizer и получения списка MV из session_data["MVs"]:
        for idx, mv in enumerate(self.session_data["MVs"]):
            min_val = mv.lower_bound  # или mv.LowerBound, если с большой буквы
            max_val = mv.upper_bound
            self.optimizer.setVecItemLimit(index=idx, min=min_val, max=max_val)

    def external_model(self, mv):
        """Вызывается оптимизатором для получения CV по MV."""
        self.last_mv = mv.copy()  # Сохраняем MV, который нужно рассчитать на клиенте
        self.cv_ready.clear()     # Сбросить флаг готовности CV
        # Ожидать, пока клиент не пришлёт CV
        if not self.cv_ready.wait(timeout=30):  # 30 секунд на ответ клиента
            raise TimeoutError("CV value not received from client in time")
        return self.current_cv.copy()  # Вернуть последнее полученное CV

    def process_iteration(self, cv_value=None):
        from BlackBoxOptimizer.BoxOptimizer.EvolutionaryOpt.EvolutionaryOpt import EvolutionaryOpt

        if isinstance(self.optimizer._CurrentOptimizerObject, EvolutionaryOpt):
            if not hasattr(self, 'evo_state'):
                pop_size = self.optimizer._CurrentOptimizerObject.population_size
                dim = len(self.session_data["MVs"])
                self.evo_state = {
                    'population': [np.random.uniform(-10, 10, dim) for _ in range(pop_size)],
                    'fitness': [None] * pop_size,
                    'current_idx': 0,
                    'generation': 0,
                    'max_generations': self.optimizer._CurrentOptimizerObject.t_max,
                    'finished': False,
                    'best_mv': None
                }

            state = self.evo_state

            # Получаем CV для текущего кандидата
            if cv_value is not None and state['current_idx'] < len(state['fitness']) and state['fitness'][state['current_idx']] is None:
                state['fitness'][state['current_idx']] = cv_value
                state['current_idx'] += 1

            # Если не все кандидаты оценены — отправляем следующий MV
            if state['current_idx'] < len(state['population']):
                mv = state['population'][state['current_idx']]
                return mv.tolist()

            # Все кандидаты оценены — делаем шаг эволюции или завершаем
          # --- Завершение: возвращаем лучший MV и больше не продолжаем
            if state['generation'] >= state['max_generations']:
                if self.session_data.get("maximize", False):
                    best_idx = int(np.argmax(state['fitness']))
                else:
                    best_idx = int(np.argmin(state['fitness']))
                state['best_mv'] = state['population'][best_idx]
                state['finished'] = True
                return []

            # --- Простейший шаг эволюции: пересоздать популяцию вокруг лучшего ---
            if self.session_data.get("maximize", False):
                best_idx = int(np.argmax(state['fitness']))
            else:
                best_idx = int(np.argmin(state['fitness']))
            best_mv = state['population'][best_idx]
            lower_bounds = np.array([mv.lower_bound for mv in self.session_data["MVs"]])
            upper_bounds = np.array([mv.upper_bound for mv in self.session_data["MVs"]])
            state['population'] = [
                np.clip(
                    best_mv + np.random.normal(0, 0.1, len(best_mv)),
                    lower_bounds,
                    upper_bounds
                )
                for _ in range(len(state['population']))
            ]
            state['fitness'] = [None] * len(state['population'])
            state['current_idx'] = 0
            state['generation'] += 1
            self.current_iteration += 1

            mv = state['population'][state['current_idx']]
            return mv.tolist()

        # --- TestStepOpt (старый код) ---
        if not hasattr(self, 'search_state'):
            mv_count = len(self.session_data["MVs"])
            current_point = self.optimizer._CurrentOptimizerObject._to_opt_model_data.vecs[:, 0].copy()
            self.search_state = {
                'current_point': current_point,
                'coord_index': 0,
                'candidate_index': 0,
                'candidates': [],
                'candidate_values': [],
                'iteration': 0
            }
            step = getattr(self.optimizer._CurrentOptimizerObject, 'step', 1.0)
            point = self.search_state['current_point']
            i = self.search_state['coord_index']
            candidates = [
                point.copy(),
                point.copy(),
                point.copy()
            ]
            candidates[0][i] -= step
            candidates[2][i] += step
            self.search_state['candidates'] = candidates
            self.search_state['candidate_values'] = [None, None, None]
            print(f"[SERVER] Инициализация поиска: MV={point}, step={step}")

        state = self.search_state

        if cv_value is not None:
            print(f"[SERVER] Получен CV={cv_value:.6f} для кандидата MV={state['candidates'][state['candidate_index']]}")
            state['candidate_values'][state['candidate_index']] = cv_value

            # История (опционально)
            if hasattr(self.optimizer._CurrentOptimizerObject, 'history_to_opt_model_data'):
                self.optimizer._CurrentOptimizerObject.history_to_opt_model_data.append(
                    state['candidates'][state['candidate_index']].copy()
                )
            if hasattr(self.optimizer._CurrentOptimizerObject, 'history_from_model_data'):
                self.optimizer._CurrentOptimizerObject.history_from_model_data.append(cv_value)

            state['candidate_index'] += 1

        if state['candidate_index'] < len(state['candidates']):
            mv = state['candidates'][state['candidate_index']]
            print(f"[SERVER] Отправляю кандидата #{state['candidate_index']} по координате {state['coord_index']}: MV={mv}")
            return mv.tolist()

        # ВЫБОР ЛУЧШЕГО КАНДИДАТА ПО maximize/minimize
        if self.session_data.get("maximize", False):
            best_idx = int(np.argmax(state['candidate_values']))
        else:
            best_idx = int(np.argmin(state['candidate_values']))

        print(f"[SERVER] Лучший кандидат по координате {state['coord_index']}: MV={state['candidates'][best_idx]}, CV={state['candidate_values'][best_idx]:.6f}")
        state['current_point'] = state['candidates'][best_idx].copy()
        state['coord_index'] += 1

        mv_count = len(self.session_data["MVs"])

        if state['coord_index'] >= mv_count:
            # Обновляем vecs через property (первый кандидат)
            self.optimizer._CurrentOptimizerObject._to_opt_model_data._vec[:, self.optimizer._CurrentOptimizerObject._to_opt_model_data.values_index_start] = state['current_point']
            self.current_iteration += 1
            print(f"[SERVER] Итерация оптимизации завершена. Новая точка: {state['current_point']}, итерация #{self.current_iteration}")
            del self.search_state
            return state['current_point'].tolist()
        else:
            step = getattr(self.optimizer._CurrentOptimizerObject, 'step', 1.0)
            point = state['current_point']
            i = state['coord_index']
            candidates = [
                point.copy(),
                point.copy(),
                point.copy()
            ]
            candidates[0][i] -= step
            candidates[2][i] += step
            state['candidates'] = candidates
            state['candidate_values'] = [None, None, None]
            state['candidate_index'] = 0
            print(f"[SERVER] Переход к координате {i}. Кандидаты: {[c.tolist() for c in candidates]}")
            return candidates[0].tolist()

class RtoService(RtoApi_pb2_grpc.RtoServiceServicer):
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        print("Сервер RTO инициализирован и готов к работе")

    def GetServiceStatus(self, request, context):
        return RtoApi_pb2.ServiceStatus(
            serviceId="",
            status="Running",
            serviceType="RtoService",
            user="",
            message=""
        )

    def StartOptimizeSession(self, request, context):
        try:
            session_id = str(uuid.uuid4())
            session_data = {
                "CVs": request.cvs,
                "MVs": request.mvs,
                "maximize": request.maximize,
                "optimization_method": request.optimization_method,
                "max_iterations": request.max_iterations,
                "model_id": request.model_id
            }
            
            with self.lock:
                self.sessions[session_id] = OptimizationSession(session_id, session_data)
                print(f"Created new session: {session_id}")

            return RtoApi_pb2.StartResponse(
                optimization_instance_id=session_id,
                is_good=True,
                message="Session created successfully"
            )
        except Exception as e:
            print(f"Error creating session: {str(e)}")
            return RtoApi_pb2.StartResponse(
                optimization_instance_id="",
                is_good=False,
                message=str(e)
            )

    def OptimizeIteration(self, request, context):
        try:
            session_id = request.optimization_instance_id
            if session_id not in self.sessions:
                return RtoApi_pb2.OptimizeIterationResponse(
                    mv_values=[],
                    sessionId="",
                    flag=-1,
                    message="err_sessionNotFound"
                )

            session = self.sessions[session_id]

            # ВСЕГДА вызываем process_iteration, даже если cv не пришёл
            if request.HasField('objective_function_value'):
                mv = session.process_iteration(request.objective_function_value.numericValue)
            else:
                mv = session.process_iteration(None)

            mv_values = [
                RtoApi_pb2.TagVal(
                    tagId=session.session_data["MVs"][i].id,
                    numericValue=mv[i],
                    isGood=True
                ) for i in range(len(mv))
            ]

            is_completed = session.current_iteration >= session.session_data["max_iterations"]

            return RtoApi_pb2.OptimizeIterationResponse(
                mv_values=mv_values,
                sessionId=session_id,
                flag=3 if is_completed else 0,
                message="Optimization completed" if is_completed else "In progress"
            )

        except Exception as e:
            print(f"Optimization iteration error: {str(e)}")
            return RtoApi_pb2.OptimizeIterationResponse(
                mv_values=[],
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
    RtoApi_pb2_grpc.add_RtoServiceServicer_to_server(RtoService(), server)
    server.add_insecure_port('[::]:5081')
    server.start()
    print("Сервер запущен и слушает порт 5081")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()