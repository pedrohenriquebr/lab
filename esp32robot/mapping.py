import numpy as np
import cv2

# ==========================================
# CONFIGURAÇÃO COLAB (MUDE PARA False se quiser debug local)
# ==========================================
import gymnasium as gym
import numpy as np
import random

class SimpleMapper:
    def __init__(self, map_size=600, scale=60):
        """
        map_size: Tamanho da imagem do mapa em pixels (ex: 600x600)
        scale: Pixels por metro (ex: 20px = 1 metro). Aumente para dar zoom.
        """
        self.map_size = map_size
        self.scale = scale
        self.center = map_size // 2
        
        # Inicializa mapa cinza (Desconhecido)
        self.grid = np.full((map_size, map_size), 127, dtype=np.uint8)
        
        # Posição do robô no mundo (x, y, angulo)
        self.pose = [0, 0, 0] 

    def update_pose(self, action):
        """Atualiza posição estimada baseada no comando (Dead Reckoning)"""
        # AJUSTE ESSES VALORES: Velocidade estimada do seu robô
        SPEED = 0.15      # Metros por passo
        ROT_SPEED = 0.25  # Radianos por passo
        
        x, y, theta = self.pose
        
        if action == 1: # FWD
            x += SPEED * np.cos(theta)
            y += SPEED * np.sin(theta)
        elif action == 2: # BACK
            x -= SPEED * np.cos(theta)
            y -= SPEED * np.sin(theta)
        elif action == 3: # LEFT
            theta -= ROT_SPEED
        elif action == 4: # RIGHT
            theta += ROT_SPEED
            
        self.pose = [x, y, theta]

    def update_map(self, depth_img):
        """Usa a imagem de profundidade para pintar o mapa"""
        h, w = depth_img.shape
        
        # Pegamos uma linha "no horizonte" (ex: 40% da altura) 
        # para evitar pegar o chão, mas pegar as paredes.
        scan_row = int(h * 0.6) 
        scan_line = depth_img[scan_row, :]
        
        # Reduz resolução para não processar todos os pixels (pega 1 a cada 4)
        step = 4
        
        # Campo de Visão (FOV) da ESP32-CAM (~60 graus)
        fov = np.radians(60)
        rx, ry, r_theta = self.pose
        
        start_angle = r_theta - (fov / 2)
        angle_inc = fov / (w / step)
        
        # Coordenada do robô no mapa (pixels)
        px_r = int(self.center + rx * self.scale)
        py_r = int(self.center + ry * self.scale)

        for i in range(0, w, step):
            intensity = scan_line[i]
            
            # 1. CONVERTE BRILHO EM DISTÂNCIA (Metros)
            # Calibração empírica para MiDaS:
            if intensity < 10: 
                dist = 4.0 # Infinito/Longe
            else:
                # Quanto mais brilhante, mais perto.
                # Ajuste o 50.0 conforme necessário para casar com a realidade
                dist = 50.0 / (intensity + 0.1)
            
            # Se for muito longe, ignoramos para não sujar o mapa
            if dist > 3.0: continue

            # 2. CALCULA POSIÇÃO DO OBSTÁCULO (Trigonometria)
            angle = start_angle + (i/step) * angle_inc
            
            ox = rx + dist * np.cos(angle)
            oy = ry + dist * np.sin(angle)
            
            px_o = int(self.center + ox * self.scale)
            py_o = int(self.center + oy * self.scale)
            
            # Garante que está dentro da imagem
            if 0 <= px_o < self.map_size and 0 <= py_o < self.map_size:
                # Desenha LINHA BRANCA (Livre) do robô até o obstáculo
                cv2.line(self.grid, (px_r, py_r), (px_o, py_o), 255, 1)
                
                # Desenha PONTO PRETO (Ocupado) no fim
                # Só marcamos obstáculo se a distância for confiável (< 2m)
                if dist < 2.0:
                    cv2.circle(self.grid, (px_o, py_o), 5, 0, -1)

    def get_display_img(self):
        """Retorna o mapa com o robô desenhado (sem alterar o grid original)"""
        display = cv2.cvtColor(self.grid, cv2.COLOR_GRAY2BGR)
        
        # Desenha o robô (Seta Vermelha)
        rx, ry, theta = self.pose
        px = int(self.center + rx * self.scale)
        py = int(self.center + ry * self.scale)
        
        # Ponta da seta
        tip_x = int(px + 10 * np.cos(theta))
        tip_y = int(py + 10 * np.sin(theta))
        
        cv2.arrowedLine(display, (px, py), (tip_x, tip_y), (0, 0, 255), 2)
        return display