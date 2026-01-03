import cv2 
import requests
import numpy as np
import torch
import os
import time
from collections import deque
from functools import wraps
from typing import Callable

from esp32robot.agents import DQN
from esp32robot.mapping import SimpleMapper
from esp32robot.vision import VisionProcessor

history_executions: dict[str,list[float]] = {}

def debug_time(history_dictionary:None | dict[str,list[float]]=None):
    def debug_time_decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kargs):
            t1 = time.time()
            result = func(*args,**kargs)
            t2 = time.time() - t1
            func_name = func.__name__
            if history_dictionary is not None:
                if func_name not in history_dictionary.keys():
                    history_dictionary[func_name] = []
                history_dictionary[func_name].append(t2)
            # print(f'[TIME] {func_name} -> {t2 * 1000:.1f}ms')
            return result
        return wrapper
    return debug_time_decorator

class DeployedAgent:
    def __init__(self, model_path, epsilon=0.0, is_2d_model=True):
        self.device = torch.device("cpu")
        self.epsilon = epsilon
        self.is_2d_model = is_2d_model

        # --- CONFIGURAÇÃO DO STACKING ---
        self.stack_size = 8
        self.num_sensors = 3
        self.input_size = self.num_sensors * self.stack_size # 24
        
        # Buffer de observação (Memória de Curto Prazo)
        self.obs_stack = deque([np.zeros(self.num_sensors)] * self.stack_size, maxlen=self.stack_size)

        print(f"🧠 Configurando Agente com Frame Stacking ({self.stack_size} frames)")
        self.policy_net = DQN(self.input_size, 5).to(self.device)
        
        self.mapper = SimpleMapper(map_size=400, scale=60)
        self.last_action = 0 

        # Carrega Pesos
        if os.path.exists(model_path):
            print(f"✅ Carregando pesos de: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint)
            self.policy_net.eval()
        else:
            print(f"❌ ERRO: Modelo {model_path} não encontrado!")
            
        self.vision = VisionProcessor(width=150, height=150)
    
    @debug_time(history_dictionary=history_executions)
    def process_frame(self, img_bgr):
        # Redimensiona
        img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_180)
        img_resized = cv2.resize(img_rotated, (150, 150))

        # Profundidade e Sensores Atuais (3 valores)
        depth = self.vision.depth_estimator.predict(img_resized)
        current_sensors = self.vision.compute_state_2d(depth)
        
        # --- ATUALIZA MAPA ---
        self.mapper.update_pose(self.last_action)
        self.mapper.update_map(depth)
        
        # --- ATUALIZA STACK ---
        self.obs_stack.append(np.array(current_sensors, dtype=np.float32))
        
        # Gera estado final (12 valores)
        state_stacked = np.concatenate(self.obs_stack)

        # Cria Debug Visual
        debug_img = self._create_debug_2d(img_resized, depth, current_sensors)
        
        return state_stacked, debug_img, current_sensors

    @debug_time(history_dictionary=history_executions)
    def _create_debug_2d(self, img, depth, sensors):
        """Visualização com Linha de Corte e Mapa"""
        
        # 1. Prepara cores
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
        if depth_color.shape[:2] != img.shape[:2]: 
            depth_color = cv2.resize(depth_color, (img.shape[1], img.shape[0]))
            
        # 2. Junta
        combined_small = np.hstack((img, depth_color))
        
        # 3. Amplia
        SCALE = 3
        final_w = combined_small.shape[1] * SCALE
        final_h = combined_small.shape[0] * SCALE
        debug_view = cv2.resize(combined_small, (final_w, final_h), interpolation=cv2.INTER_NEAREST)
        
        # Linha de Corte
        roi_ratio = 0.7 
        y_cut = int(img.shape[0] * roi_ratio * SCALE)
        
        # Sombra
        overlay = debug_view.copy()
        cv2.rectangle(overlay, (0, y_cut), (final_w, final_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, debug_view, 0.4, 0, debug_view)
        
        # Linha Amarela
        cv2.line(debug_view, (0, y_cut), (final_w, y_cut), (0, 255, 255), 2)

        # Sensores
        w_panel = final_w // 2
        h_panel = final_h
        step_x = w_panel // 3
        
        line_color = (0, 255, 0)
        for offset in [0, w_panel]:
            cv2.line(debug_view, (offset + step_x, 0), (offset + step_x, y_cut), line_color, 2)
            cv2.line(debug_view, (offset + 2*step_x, 0), (offset + 2*step_x, y_cut), line_color, 2)

        # Valores
        for i, val in enumerate(sensors):
            txt = f"{val:.2f}"
            color = (0, 0, 255) if val < 0.3 else (0, 255, 0)
            
            cx = i * step_x + 20 
            cy = y_cut - 20
            
            # Esquerda
            cv2.putText(debug_view, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 5)
            cv2.putText(debug_view, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            # Direita
            cx2 = w_panel + cx
            cv2.putText(debug_view, txt, (cx2, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 5)
            cv2.putText(debug_view, txt, (cx2, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Adiciona Mapa
        map_img = self.mapper.get_display_img()
        map_h = h_panel
        map_w = h_panel 
        map_resized = cv2.resize(map_img, (map_w, map_h), interpolation=cv2.INTER_NEAREST)
        
        return np.hstack((debug_view, map_resized))
    
    @debug_time(history_dictionary=history_executions)
    def get_action(self, state, return_q_values=False):
        q_values = None
        # Epsilon-Greedy (mesmo em deploy, às vezes 0.05 ajuda a destravar)
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, 5)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            action = int(np.argmax(q_values))
        
        self.last_action = action
        if return_q_values:
            return action, q_values
        return action

class RealRobotController:
    def __init__(self, commands, stream_url):
        self.commands = commands
        self.stream_url = stream_url
    
    @debug_time(history_dictionary=history_executions)
    def send_command(self, action):
        try:
            url = self.commands[action]
            headers = {"ngrok-skip-browser-warning": "true", "Connection": "close"}
            response = requests.get(url, headers=headers, timeout=0.5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Erro comando: {e}")
            return False
    
    @debug_time(history_dictionary=history_executions)
    def get_frame(self):
        try:
            r = requests.get(self.stream_url, stream=True, timeout=5)
            if r.status_code == 200:
                bytes_d = b''
                for chunk in r.iter_content(chunk_size=1024):
                    bytes_d += chunk
                    a = bytes_d.find(b'\xff\xd8')
                    b = bytes_d.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_d[a:b+2]
                        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            return None
        except: return None