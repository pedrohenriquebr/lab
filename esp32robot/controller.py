# ==========================================
# 1. IMPORTS E CONFIGURAÇÃO
# ==========================================
import cv2 
import requests
import cv2
import numpy as np
import torch
import os
import numpy as np
import cv2
import os
import torch
import numpy as np

from esp32robot.agents import DQN
from esp32robot.mapping import SimpleMapper
from esp32robot.vision import VisionProcessor


# para rastrear os tempos de execução de cada função e com isso identificar gargalos

history_executions: dict[str,list[float]] = {}

from functools import wraps
from typing import Callable
import numpy as np


def debug_time(history_dictionary:None | dict[str,list[float]]=None):
    def debug_time_decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kargs):
            import time
            t1 = time.time()
            result = func(*args,**kargs)
            t2 = time.time() - t1
            func_name = func.__name__
            if history_dictionary is not None:
                if func_name not in history_dictionary.keys():
                    history_dictionary[func_name] = []
                history_dictionary[func_name].append(t2)
            print(f'[TIME] function {func.__name__} -> elapsed {t2 * 1000}ms')
            return result
        return wrapper
    return debug_time_decorator

# Criar uma instância do agente com os MESMOS parâmetros do treino
# (Assumindo que você usou DQN)
class DeployedAgent:
    def __init__(self, model_path, epsilon=0.0, is_2d_model=True):
        self.device = torch.device("cpu")
        self.epsilon = epsilon
        self.is_2d_model = is_2d_model

        # 🔴 AJUSTE DINÂMICO DA REDE
        if self.is_2d_model:
            print("🧠 Configurando Agente para Modelo 2D (3 Inputs)")
            input_size = 3
        else:
            print("🧠 Configurando Agente para Modelo Voxel (9 Inputs)")
            input_size = 9

        self.policy_net = DQN(input_size, 5).to(self.device)
        self.mapper = SimpleMapper(map_size=400, scale=30)
        self.last_action = 0 # Guarda a última ação para estimar movimento

        # Carrega Pesos
        full_path = model_path
        if os.path.exists(full_path):
            print(f"✅ Carregando pesos de: {full_path}")
            checkpoint = torch.load(full_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint)
            self.policy_net.eval()
        else:
            print(f"❌ ERRO: Modelo {full_path} não encontrado!")
            # Não damos raise error pra permitir testar sem modelo se quiser
            
        self.vision = VisionProcessor(width=450, height=300)
    
    
    def print_model_size(self):
        """Calcula e imprime o tamanho da rede neural em memória"""
        param_size = 0
        for param in self.policy_net.parameters():
            param_size += param.nelement() * param.element_size()
            
        buffer_size = 0
        for buffer in self.policy_net.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        size_all_kb = (param_size + buffer_size) / 1024
        
        print(f"📊 Estatísticas de Memória da Rede Neural:")
        print(f"   - Parâmetros totais: {sum(p.numel() for p in self.policy_net.parameters())}")
        print(f"   - Tamanho em Bytes: {param_size + buffer_size} bytes")
        print(f"   - Tamanho em KB: {size_all_kb:.2f} KB")
        print(f"   - Tamanho em MB: {size_all_mb:.6f} MB")
        
    @debug_time(history_dictionary=history_executions)
    def process_frame(self, img_bgr):
        # Redimensiona
        img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_180)
        img_resized = cv2.resize(img_rotated, (450, 300))

        # Profundidade
        depth = self.vision.depth_estimator.predict(img_resized)
        
        # --- NOVO: ATUALIZA O MAPA ---
        # 1. Atualiza posição baseado no que fizemos no passo anterior
        self.mapper.update_pose(self.last_action)
        # 2. Atualiza obstáculos baseado no que estamos vendo agora
        self.mapper.update_map(depth)
        
        # 🔴 LÓGICA HÍBRIDA
        # Se for modelo 2D, calcula os 3 sensores.
        if self.is_2d_model:
            state = self.vision.compute_state_2d(depth)
        else:
            # Lógica antiga (9 estados)
            mask_t = self.vision.filter_target(img_resized)
            mask_o = (depth > 150).astype(np.uint8) * 255
            state = self.vision.compute_state(mask_t, mask_o, depth) # Método antigo teria que existir

        # Cria Debug
        debug_img = self._create_debug_2d(img_resized, depth, state)
        
        return state, debug_img

    @debug_time(history_dictionary=history_executions)
    def _create_debug_2d(self, img, depth, state):
        """Visualização com LINHA DE CORTE (ROI) para ver o chão ignorado"""
        
        # 1. Prepara as cores
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
        if depth_color.shape[:2] != img.shape[:2]: 
            depth_color = cv2.resize(depth_color, (img.shape[1], img.shape[0]))
            
        # 2. Junta
        combined_small = np.hstack((img, depth_color))
        
        # 3. Amplia (SCALE)
        SCALE = 3
        final_w = combined_small.shape[1] * SCALE
        final_h = combined_small.shape[0] * SCALE
        debug_view = cv2.resize(combined_small, (final_w, final_h), interpolation=cv2.INTER_NEAREST)
        
        # --- NOVO: DESENHAR A LINHA DE CORTE (ROI) ---
        # Calculamos onde fica o 70% na imagem ampliada
        roi_ratio = 0.7 
        y_cut = int(img.shape[0] * roi_ratio * SCALE) # Altura original * 0.7 * Escala
        
        # A. Escurecer a área ignorada (Opcional, mas fica ótimo)
        # Cria uma cópia para fazer transparência
        overlay = debug_view.copy()
        # Desenha retangulo preto na parte de baixo
        cv2.rectangle(overlay, (0, y_cut), (final_w, final_h), (0, 0, 0), -1)
        # Mistura com a original (efeito de sombra)
        alpha = 0.6 # Opacidade da sombra
        cv2.addWeighted(overlay, alpha, debug_view, 1 - alpha, 0, debug_view)
        
        # B. Desenha a Linha Divisória
        cv2.line(debug_view, (0, y_cut), (final_w, y_cut), (0, 255, 255), 2) # Linha Amarela
        cv2.putText(debug_view, "IGNORADO (CHAO)", (10, final_h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)

        # --- RESTO DO DESENHO (Igual ao anterior) ---
        w_panel = final_w // 2
        h_panel = final_h
        step_x = w_panel // 3
        
        # Linhas verticais dos sensores
        line_color = (0, 255, 0)
        for offset in [0, w_panel]:
            # Só desenha as linhas verticais ATÉ a linha de corte (y_cut)
            # Para mostrar que o sensor só lê até ali
            cv2.line(debug_view, (offset + step_x, 0), (offset + step_x, y_cut), line_color, 2)
            cv2.line(debug_view, (offset + 2*step_x, 0), (offset + 2*step_x, y_cut), line_color, 2)

        # Escreve valores
        for i, val in enumerate(state):
            txt = f"{val:.2f}"
            color = (0, 0, 255) if val < 0.3 else (0, 255, 0)
            
            cx = i * step_x + 20 
            cy = y_cut - 20 # Desenha o texto logo acima da linha de corte
            
            # Texto Esquerda
            cv2.putText(debug_view, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 5)
            cv2.putText(debug_view, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Texto Direita
            cx2 = w_panel + cx
            cv2.putText(debug_view, txt, (cx2, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 5)
            cv2.putText(debug_view, txt, (cx2, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        map_img = self.mapper.get_display_img()
         
        # Vamos redimensionar o mapa para ter a mesma altura do debug_view
        h_debug, w_debug = debug_view.shape[:2]
        
        # Redimensiona mapa mantendo proporção quadrada
        map_h = h_debug
        map_w = h_debug 
        map_resized = cv2.resize(map_img, (map_w, map_h), interpolation=cv2.INTER_NEAREST)
        
        # Junta tudo: [ Câmera | Depth | Mapa ]
        final_composition = np.hstack((debug_view, map_resized))
        
        return final_composition
    
    @debug_time(history_dictionary=history_executions)
    def get_action(self, state):
        # Converte estado para tensor
        # State é tupla (0.5, 0.8, 0.2)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)

        state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        action  = int(np.argmax(q_values))
        
        self.last_action = action
        return action
# ==========================================
# 3. CONTROLLER DO ROBÔ REAL
# ==========================================
class RealRobotController:
    def __init__(self, commands, stream_url):
        self.commands = commands
        self.stream_url = stream_url
    
    @debug_time(history_dictionary=history_executions)
    def send_command(self, action):
        """Envia comando HTTP para o ESP32"""
        try:
            url = self.commands[action]
            headers = {
                "ngrok-skip-browser-warning": "true",
                "User-Agent": "RobotAgent/1.0", # Identidade curta
                "Connection": "close" # Evita manter conexão aberta acumulando lixo
            }
            
            response = requests.get(url,headers=headers,timeout=5)
            print(f"📡 Ação {action} -> {url.split('/')[-1]} | Status: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Falha ao enviar comando: {e}")
            return False
    
    @debug_time(history_dictionary=history_executions)
    def get_frame(self):
        """Captura frame da ESP32-CAM"""
        try:
            response = requests.get(self.stream_url, stream=True, timeout=15)
            if response.status_code == 200:
                frame_buffer = b''
                for chunk in response.iter_content(chunk_size=1024):
                    frame_buffer += chunk
                    start = frame_buffer.find(b'\xff\xd8')
                    end = frame_buffer.find(b'\xff\xd9')
                    
                    if start != -1 and end != -1:
                        jpeg_frame = frame_buffer[start:end+2]
                        frame_buffer = frame_buffer[end+2:]
                        
                        # Converte para OpenCV
                        nparr = np.frombuffer(jpeg_frame, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        return img
            return None
        except Exception as e:
            print(f"❌ Erro no stream: {e}")
            return None
