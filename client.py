import grpc
import RtoApi_pb2
import RtoApi_pb2_grpc
import time
import uuid
from typing import List, Dict, Optional
import cmd
import json
import shlex

class RTOClient:
    def __init__(self, host: str = 'localhost:5081'):
        self.channel = grpc.insecure_channel(host)
        self.stub = RtoApi_pb2_grpc.DataProcessingServiceStub(self.channel)
    
    def get_service_status(self) -> Dict:
        response = self.stub.GetServiceStatus(RtoApi_pb2.ArgRequest(accessToken=""))
        return {
            "serviceId": response.serviceId,
            "status": response.status,
            "serviceType": response.serviceType,
            "user": response.user,
            "message": response.message
        }
    
    def start(
        self,
        cvs: List[Dict[str, any]],
        mvs: List[Dict[str, any]],
        objective_function: str,
        maximize: bool,
        optimization_method: str = "GradientDescent",  # "GradientDescent" или "Evolutionary"
        max_iterations: int = 100,  # Количество итераций для обоих методов
        model_id: str = ""
    ) -> str:
        """Создает новую сессию оптимизации с выбранным методом"""
        # Генерация ID для переменных
        for cv in cvs:
            cv['id'] = str(uuid.uuid4())
        for mv in mvs:
            mv['id'] = str(uuid.uuid4())
        
        # Заменяем имена переменных в целевой функции на их ID
        modified_objective = objective_function
        for cv in cvs:
            modified_objective = modified_objective.replace(f"[{cv['name']}]", f"[{cv['id']}]")
        for mv in mvs:
            modified_objective = modified_objective.replace(f"[{mv['name']}]", f"[{mv['id']}]")
        
        # Формируем запрос
        request = RtoApi_pb2.StartRequest(
            accessToken="",
            cvs=[RtoApi_pb2.TagType(
                id=cv["id"],
                name=cv["name"],
                dataType=cv.get("dataType", "Numeric"),
                lower_bound=cv["lower_bound"],
                upper_bound=cv["upper_bound"])
                for cv in cvs
            ],
            mvs=[RtoApi_pb2.TagType(
                id=mv["id"],
                name=mv["name"],
                dataType=mv.get("dataType", "Numeric"),
                lower_bound=mv["lower_bound"],
                upper_bound=mv["upper_bound"])
                for mv in mvs
            ],
            objective_function=modified_objective,
            maximize=maximize,
            optimization_method=optimization_method,
            max_iterations=max_iterations,  # Передаем количество итераций
            model_id=model_id
        )
        
        # Отправляем запрос
        response = self.stub.Start(request)
        if not response.is_good:
            raise RuntimeError(f"Ошибка создания сессии: {response.message}")
        return response.optimization_instance_id
    
    def start_optimize(
        self,
        session_id: str,
        initial_cvs: Optional[List[Dict[str, any]]] = None,
        initial_mvs: Optional[List[Dict[str, any]]] = None
    ) -> int:
        # Преобразуем initial_mvs в формат для gRPC
        grpc_mvs = []
        if initial_mvs:
            grpc_mvs = [RtoApi_pb2.TagVal(
                tagId=mv["id"],
                numericValue=mv["value"],
                timeStamp=int(time.time()),
                isGood=True)
                for mv in initial_mvs
            ]
        
        response = self.stub.StartOptimize(RtoApi_pb2.StartOptimizeRequest(
            accessToken="",
            optimization_instance_id=session_id,
            initial_mvs=grpc_mvs
        ))
        return response.flag
    
    def pause(self, session_id: str) -> bool:
        response = self.stub.Pause(RtoApi_pb2.PauseRequest(
            accessToken="",
            sessionId=session_id
        ))
        return response.is_paused
    
    def stop(self, session_id: str) -> bool:
        response = self.stub.Stop(RtoApi_pb2.StopRequest(
            accessToken="",
            sessionId=session_id
        ))
        return "success" in response.message.lower()
    
    def close(self):
        self.channel.close()

class RTOCmd(cmd.Cmd):
    prompt = 'RTO> '
    
    def __init__(self):
        super().__init__()
        self.client = RTOClient()
        self.sessions = {}
    
    def do_status(self, arg):
        """Проверка статуса сервиса: status"""
        status = self.client.get_service_status()
        print(f"Статус сервиса: {status['status']}")
        print(f"Тип сервиса: {status['serviceType']}")
        print(f"Сообщение: {status['message']}")
    
    def do_start(self, arg):
        """Создание новой сессии оптимизации"""
        try:
            args = self.parse_args(arg)
            
            if not all(k in args for k in ['cvs', 'mvs', 'obj', 'max']):
                print("Не хватает обязательных параметров: -cvs, -mvs, -obj, -max")
                return
            
            cvs = json.loads(args['cvs'])
            mvs = json.loads(args['mvs'])
            objective_function = args['obj']
            maximize = args['max'].lower() == 'true'
            
            optimization_method = args.get('method', "Gradient Descent")
            max_iterations = int(args.get('iter', 100))
            model_id = args.get('model', "")
            
            session_id = self.client.start(
                cvs=cvs,
                mvs=mvs,
                objective_function=objective_function,
                maximize=maximize,
                optimization_method=optimization_method,
                max_iterations=max_iterations,
                model_id=model_id
            )
            
            self.sessions[session_id] = {
                'cvs': cvs,
                'mvs': mvs,
                'objective_function': objective_function,
                'maximize': maximize
            }
            
            print(f"Создана сессия с ID: {session_id}")
            print(f"Файл сессии создан: session_{session_id}.txt")
            
        except Exception as e:
            print(f"Ошибка: {str(e)}")
    
    def do_optimize(self, arg):
        """Запуск оптимизации для сессии"""
        try:
            args = self.parse_args(arg)
            
            if 'id' not in args:
                print("Не указан ID сессии (-id)")
                return
            
            session_id = args['id']
            
            if session_id not in self.sessions:
                print(f"Сессия {session_id} не найдена")
                return
            
            initial_cvs = []
            if 'cvs' in args:
                cvs_data = json.loads(args['cvs'])
                initial_cvs = [{
                    'id': next((cv['id'] for cv in self.sessions[session_id]['cvs'] if cv['name'] == cv_data['name']), None),
                    'value': cv_data['value']
                } for cv_data in cvs_data if 'name' in cv_data and 'value' in cv_data]
            
            initial_mvs = []
            if 'mvs' in args:
                mvs_data = json.loads(args['mvs'])
                initial_mvs = [{
                    'id': next((mv['id'] for mv in self.sessions[session_id]['mvs'] if mv['name'] == mv_data['name']), None),
                    'value': mv_data['value']
                } for mv_data in mvs_data if 'name' in mv_data and 'value' in mv_data]
            
            flag = self.client.start_optimize(
                session_id=session_id,
                initial_cvs=initial_cvs,
                initial_mvs=initial_mvs
            )
            
            statuses = {
                0: "Оптимизация успешно запущена",
                1: "Оптимизация возобновлена после паузы", 
                2: "Оптимизация уже выполняется"
            }
            
            print(statuses.get(flag, f"Неизвестный статус: {flag}"))
            
        except Exception as e:
            print(f"Ошибка: {str(e)}")
    
    def do_pause(self, arg):
        """Приостановка сессии: pause -id <session_id>"""
        try:
            args = self.parse_args(arg)
            
            if 'id' not in args:
                print("Не указан ID сессии (-id)")
                return
            
            session_id = args['id']
            
            if self.client.pause(session_id):
                print(f"Сессия {session_id} приостановлена")
            else:
                print(f"Не удалось приостановить сессию {session_id}")
        except Exception as e:
            print(f"Ошибка: {str(e)}")
    
    def do_stop(self, arg):
        """Остановка сессии: stop -id <session_id>"""
        try:
            args = self.parse_args(arg)
            
            if 'id' not in args:
                print("Не указан ID сессии (-id)")
                return
            
            session_id = args['id']
            
            if self.client.stop(session_id):
                print(f"Сессия {session_id} остановлена")
                if session_id in self.sessions:
                    del self.sessions[session_id]
            else:
                print(f" {session_id}")
        except Exception as e:
            print(f"Ошибка: {str(e)}")
    
    def do_list(self, arg):
        """Список всех сессий: list"""
        if not self.sessions:
            print("Нет активных сессий")
            return
            
        print("Активные сессии:")
        for session_id, data in self.sessions.items():
            print(f"ID: {session_id}")
            print(f"  Целевая функция: {data['objective_function']}")
            print(f"  Максимизация: {data['maximize']}")
            
            if 'cvs' in data and isinstance(data['cvs'], list):
                print(f"  CVs: {', '.join([cv.get('name', '?') for cv in data['cvs']])}")
            
            if 'mvs' in data and isinstance(data['mvs'], list):
                print(f"  MVs: {', '.join([mv.get('name', '?') for mv in data['mvs']])}")
            
            print()
    
    def do_exit(self, arg):
        """Выход из программы: exit"""
        self.client.close()
        print("Выход")
        return True
    
    def parse_args(self, arg_str):
        """Парсит аргументы командной строки"""
        args = {}
        parts = shlex.split(arg_str)
        
        i = 0
        while i < len(parts):
            if parts[i].startswith('-'):
                key = parts[i][1:]
                if i + 1 < len(parts) and not parts[i+1].startswith('-'):
                    args[key] = parts[i+1]
                    i += 2
                else:
                    args[key] = ""
                    i += 1
            else:
                i += 1
                
        return args

if __name__ == '__main__':
    RTOCmd().cmdloop()