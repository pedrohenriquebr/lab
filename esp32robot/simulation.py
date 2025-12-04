import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random

class Esp322DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.window_size = 600  # Tamanho da janela (pixels)
        self.map_scale = 100    # 100 pixels = 1 metro (Mundo de 6x6 metros)
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Action Space: 0:Stop, 1:Fwd, 2:Back, 3:Left, 4:Right
        self.action_space = spaces.Discrete(5)
        
        # Observation Space: 3 valores flutuantes (Distância Esq, Centro, Dir)
        # Normalizado entre 0 (colado) e 1 (longe/infinito)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.car_pos = np.array([3.0, 3.0]) # Centro do mapa (metros)
        self.car_angle = 0.0 # Radianos
        self.car_speed = 0.05 # Metros por passo
        self.car_radius = 0.15 # Raio do robô (15cm)
        
        self.obstacles = []
        self.sensors = [0.0, 0.0, 0.0] # Leituras atuais

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reinicia posição
        self.car_pos = np.array([3.0, 3.0])
        self.car_angle = random.uniform(0, 2 * math.pi)
        self.steps_idle = 0       # Passos que ele ficou parado/girando
        self.collision_count = 0  # Colisões (se o episódio não acabar na primeira)
        self.last_pos = self.car_pos.copy()
        
        # Gera obstáculos aleatórios (Círculos)
        self.obstacles = []
        for _ in range(8): # 8 Obstáculos
            ox = random.uniform(0.5, 5.5)
            oy = random.uniform(0.5, 5.5)
            # Evita nascer em cima do carro
            if np.linalg.norm([ox-3, oy-3]) > 1.0:
                self.obstacles.append({'pos': np.array([ox, oy]), 'radius': 0.3})
        
        # Paredes (Bordas do mundo) são tratadas no raycast
        
        self._update_sensors()
        return self._get_obs(), {}

    def step(self, action):
        # --- Física Simples ---
        if action == 1: # FWD
            self.car_pos[0] += self.car_speed * math.cos(self.car_angle)
            self.car_pos[1] += self.car_speed * math.sin(self.car_angle)
        elif action == 2: # BACK
            self.car_pos[0] -= self.car_speed * math.cos(self.car_angle)
            self.car_pos[1] -= self.car_speed * math.sin(self.car_angle)
        elif action == 3: # LEFT
            self.car_angle -= 0.2
        elif action == 4: # RIGHT
            self.car_angle += 0.2

        # --- Detecção de Colisão ---
        collided = False
        
        # 1. Paredes
        if (self.car_pos[0] < self.car_radius or self.car_pos[0] > 6 - self.car_radius or
            self.car_pos[1] < self.car_radius or self.car_pos[1] > 6 - self.car_radius):
            collided = True
            
        # 2. Obstáculos
        for obs in self.obstacles:
            dist = np.linalg.norm(self.car_pos - obs['pos'])
            if dist < (self.car_radius + obs['radius']):
                collided = True
                break

        self._update_sensors()
        
        # distância euclidiana
        dist_moved = np.linalg.norm(self.car_pos - self.last_pos)
        if dist_moved < 0.01: # Se moveu menos de 1cm
            self.steps_idle += 1
        self.last_pos = self.car_pos.copy()

        # 2. Detectar Colisão
        if collided:
            self.collision_count = 1 # Ou += 1 se o robô não morrer na batida
        
        # --- Recompensa ---
       # Lembrando: 0.0 = Colado, 1.0 = Livre
        d_left = self.sensors[0]
        d_center = self.sensors[1]
        d_right = self.sensors[2]

        # Lógica de Estados
        # Está muito perto na frente?
        is_blocked_front = d_center < 0.25 
        # Está preso num canto? (Bloqueado na Esquerda E na Direita)
        is_in_corner = (d_left < 0.25) and (d_right < 0.25)

        # --- RECOMPENSA ---
        reward = 0
        terminated = False

        if collided:
            reward = -1.0
            terminated = True
        else:
            # === LÓGICA DE PRIORIDADE ===
            
            # 1. SITUAÇÃO CRÍTICA: CANTO / BECO SEM SAÍDA
            if is_in_corner:
                # Se está num canto, PROÍBE girar e andar pra frente
                if action == 2: # BACK
                    reward += 0.5 # SALVAÇÃO!
                else:
                    # Qualquer outra coisa (Frente, Girar, Parar) é punida severamente
                    # para forçar ele a aprender que SÓ a ré funciona aqui.
                    reward -= 0.8 

            # 2. SITUAÇÃO DE PERIGO FRONTAL (Parede na frente, mas lados livres)
            elif is_blocked_front:
                if action == 2: # BACK
                    reward += 0.2
                elif action in [3, 4]: # GIRAR
                    reward += 0.1 # Girar é aceitável aqui para desviar
                elif action == 1: # FWD
                    reward -= 1.0 # Suicídio
                else:
                    reward -= 0.1

            # 3. SITUAÇÃO NORMAL (Navegação)
            else:
                if action == 1: # FWD
                    reward += 0.2
                elif action == 0: # STOP
                    reward -= 0.05
                elif action in [3, 4]: # TURNS
                    reward -= 0.05 # Custo pequeno para não ficar dançando
                elif action == 2: # BACK
                    reward -= 0.1 # Ré sem motivo é ruim

            
            if dist_moved < 0.01:
                reward -= 0.1
            # 4. PENALIDADE POR PROXIMIDADE GERAL (Safety Margin)
            # Isso ajuda ele a não andar "raspando" na parede
            min_reading = min(self.sensors)
            if min_reading < 0.3:
                reward -= (0.3 - min_reading) * 1.0


        if self.render_mode == "human":
            self.render()

        dist_from_start = np.linalg.norm(self.car_pos - np.array([3.0, 3.0]))
        
        info = {
            "is_idle": dist_moved < 0.01,
            "collision": collided,
            "dist_traveled": dist_from_start
        }

        
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        # Retorna array numpy float32
        return np.array(self.sensors, dtype=np.float32)

    def _update_sensors(self):
        # Simula 3 raios: Esquerda (+45 graus), Centro (0), Direita (-45)
        angles = [self.car_angle - 0.78, self.car_angle, self.car_angle + 0.78]
        max_range = 3.0 # Sensor vê até 3 metros
        
        self.sensors = []
        for angle in angles:
            dist = max_range
            ray_dir = np.array([math.cos(angle), math.sin(angle)])
            
            # 1. Check Paredes (Matemática de intersecção de linha)
            # Distância para x=0, x=6, y=0, y=6
            # (Simplificado para box)
            if ray_dir[0] != 0:
                d1 = (0 - self.car_pos[0]) / ray_dir[0] if ray_dir[0] < 0 else (6 - self.car_pos[0]) / ray_dir[0]
                if 0 < d1 < dist: dist = d1
            if ray_dir[1] != 0:
                d2 = (0 - self.car_pos[1]) / ray_dir[1] if ray_dir[1] < 0 else (6 - self.car_pos[1]) / ray_dir[1]
                if 0 < d2 < dist: dist = d2
                
            # 2. Check Obstáculos (Intersecção Linha-Círculo)
            for obs in self.obstacles:
                # Vetor do carro ao centro do obstáculo
                v = obs['pos'] - self.car_pos
                # Projeção do vetor na direção do raio
                proj = np.dot(v, ray_dir)
                if proj > 0: # Só se obstáculo estiver na frente
                    closest_point_on_line = self.car_pos + proj * ray_dir
                    dist_to_center = np.linalg.norm(closest_point_on_line - obs['pos'])
                    if dist_to_center < obs['radius']:
                        # Colidiu com o círculo. Calcula distância exata
                        # Pitágoras
                        back_dist = math.sqrt(obs['radius']**2 - dist_to_center**2)
                        hit_dist = proj - back_dist
                        if 0 < hit_dist < dist:
                            dist = hit_dist

            # Normaliza (0.0 = Colado, 1.0 = Longe)
            # Vamos inverter para facilitar a rede neural (1.0 = Perto/Perigo, 0.0 = Seguro)?
            # NÃO. Vamos manter física real: Valor é Metros.
            # Mas para o obs, normalizamos 0 a 1 onde 1 é max_range
            self.sensors.append(dist / max_range)

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # Fundo Branco
        
        # Desenha Obstáculos
        for obs in self.obstacles:
            pos_px = (obs['pos'] * self.map_scale).astype(int)
            rad_px = int(obs['radius'] * self.map_scale)
            pygame.draw.circle(canvas, (100, 100, 100), pos_px, rad_px)
            
        # Desenha Carro
        car_px = (self.car_pos * self.map_scale).astype(int)
        pygame.draw.circle(canvas, (0, 0, 255), car_px, int(self.car_radius * self.map_scale))
        
        # Desenha Sensores (Raios)
        angles = [self.car_angle - 0.78, self.car_angle, self.car_angle + 0.78]
        for i, dist_norm in enumerate(self.sensors):
            dist_m = dist_norm * 3.0
            end_x = self.car_pos[0] + dist_m * math.cos(angles[i])
            end_y = self.car_pos[1] + dist_m * math.sin(angles[i])
            end_px = (np.array([end_x, end_y]) * self.map_scale).astype(int)
            
            # Cor do raio: Vermelho se perto, Verde se longe
            color = (255, 0, 0) if dist_norm < 0.2 else (0, 255, 0)
            pygame.draw.line(canvas, color, car_px, end_px, 2)

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
