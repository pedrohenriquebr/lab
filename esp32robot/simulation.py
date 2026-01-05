import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random

from pyparsing import deque

ACTIONS_INDEX = {
    'action_stop': 0,
    'action_fwd': 1,
    'action_back': 2,
    'action_left': 3,
    'action_right': 4,
    'action_turn': [3, 4]
}


class Esp322DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, env_type='default', rotation_penalty=0.1, 
                 action_weights=None, 
                 latency_steps=0,
                 stack_size=8,
                 num_sensors=3,
                 threshold_blocked_front=0.25,
                 threshold_corner=0.25,
                 threshold_proximity_danger = 0.2,
                 threshold_idle_movement = 0.05,
                 rewards =None
                 ):
        self.window_size = 600  # Tamanho da janela (pixels)
        self.map_scale = 100    # 100 pixels = 1 metro (Mundo de 6x6 metros)
        
        # Tipos: 'default', 'maze', 'circular_race_track', 'dynamic'
        self.env_type = env_type
        self.threshold_blocked_front = threshold_blocked_front
        self.threshold_corner = threshold_corner
        self.threshold_proximity_danger = threshold_proximity_danger
        self.threshold_idle_movement = threshold_idle_movement
        
        default_rewards = {
            'collision': -1.0,
            'penalities': {
                'idle': -1.0,
                'proximity_factor': -1.0,
                'indecision': -0.8
            },
           'states': {
                'in_corner': {'action_back': 0.5, 'default': -0.8},
            'blocked_front': {
                'action_back': 0.2,
                'action_turn': 0.1,
                'action_fwd': -1.0,
                'default': -3.0
            },
            'clear_path': {
                    'action_fwd': 1.0,
                    'action_stop': -2.0,
                'action_back': -5.0
            }
           }
        }
        self.rewards_config = rewards if rewards is not None else default_rewards
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.rotation_penalty = rotation_penalty
        self.action_weights = action_weights if action_weights is not None else {0:0, 1:0, 2:0, 3:0, 4:0}
        self.last_action = 0
        self.stack_size = stack_size
        self.num_sensors = num_sensors
        total_obs = self.num_sensors * self.stack_size
        self.latency_steps = latency_steps
        self.action_buffer = deque([0] * (latency_steps+1), maxlen=latency_steps+1)
        
        # Action Space: 0:Stop, 1:Fwd, 2:Back, 3:Left, 4:Right
        self.action_space = spaces.Discrete(5)
        
        # Observation Space: 3 valores flutuantes (Distância Esq, Centro, Dir)
        self.observation_space = spaces.Box(low=0, high=1, shape=(total_obs,), dtype=np.float32)
        self.obs_stack = deque([np.zeros(self.num_sensors)] * self.stack_size, maxlen=self.stack_size)


        self.car_radius = 0.15 
        self.car_speed = 0.05
        
        # Estado Inicial
        self.car_pos = np.array([3.0, 3.0])
        self.car_angle = 0.0
        
        self.obstacles = []
        self.sensors = np.zeros(self.num_sensors).tolist()

    def _spawn_car_safely(self):
        """Tenta encontrar uma posição livre para nascer"""
        max_attempts = 100
        for _ in range(max_attempts):
            # Sorteia posição longe das bordas
            x = random.uniform(0.5, 5.5)
            y = random.uniform(0.5, 5.5)
            pos = np.array([x, y])
            
            collision = False
            # Checa colisão com obstáculos
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs['pos']) < (self.car_radius + obs['radius'] + 0.2):
                    collision = True
                    break
            
            if not collision:
                return pos
        
        return np.array([3.0, 3.0]) # Fallback se falhar

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Gera o Layout do Mapa
        self.obstacles = []
        
        if self.env_type == 'maze':
            # Cria paredes usando bolinhas alinhadas (simplifica a física)
            # Paredes verticais e horizontais
            self._create_line_obstacle((1.0, 1.0), (1.0, 4.0)) # Parede Esq
            self._create_line_obstacle((5.0, 2.0), (5.0, 5.0)) # Parede Dir
            self._create_line_obstacle((2.0, 3.0), (4.0, 3.0)) # Meio
            
        elif self.env_type == 'circular_race_track':
            # Pista circular (anel externo e interno)
            # Anel Externo
            for angle in np.linspace(0, 2*math.pi, 20, endpoint=False):
                ox = 3.0 + 2.5 * math.cos(angle)
                oy = 3.0 + 2.5 * math.sin(angle)
                self.obstacles.append({'pos': np.array([ox, oy]), 'radius': 0.3, 'vel': np.array([0.0,0.0])})
            # Anel Interno
            for angle in np.linspace(0, 2*math.pi, 10, endpoint=False):
                ox = 3.0 + 1.0 * math.cos(angle)
                oy = 3.0 + 1.0 * math.sin(angle)
                self.obstacles.append({'pos': np.array([ox, oy]), 'radius': 0.3, 'vel': np.array([0.0,0.0])})

        elif self.env_type == 'dynamic':
            # Obstáculos que se mexem!
            for _ in range(6):
                ox = random.uniform(1.0, 5.0)
                oy = random.uniform(1.0, 5.0)
                vx = random.uniform(-0.03, 0.03) # Velocidade aleatória
                vy = random.uniform(-0.03, 0.03)
                self.obstacles.append({
                    'pos': np.array([ox, oy]), 
                    'radius': 0.3,
                    'vel': np.array([vx, vy]) 
                })
        else:
            # Default (Aleatório estático)
            for _ in range(8):
                ox = random.uniform(0.5, 5.5)
                oy = random.uniform(0.5, 5.5)
                self.obstacles.append({'pos': np.array([ox, oy]), 'radius': 0.3, 'vel': np.array([0.0,0.0])})

        # 2. Posiciona o Carro num lugar seguro
        self.car_pos = self._spawn_car_safely()
        self.car_angle = random.uniform(0, 2 * math.pi)
        
        # Variáveis de rastreio
        self.steps_idle = 0
        self.collision_count = 0
        self.last_pos = self.car_pos.copy()
        
        self.action_buffer.clear()
        self.action_buffer.extend([0] * (self.latency_steps+1))
        self.obs_stack.clear()
        self._update_sensors()
        initial_obs = np.array(self.sensors, dtype=np.float32)
        for _ in range(self.stack_size):
            self.obs_stack.append(initial_obs)
            
        return self._get_obs(), {}

    def _create_line_obstacle(self, start, end, radius=0.2):
        """Cria uma parede de obstáculos circulares entre dois pontos"""
        p1 = np.array(start)
        p2 = np.array(end)
        dist = np.linalg.norm(p2 - p1)
        num_obs = int(dist / (radius * 1.5)) # Densidade
        
        for i in range(num_obs + 1):
            t = i / max(1, num_obs)
            pos = p1 + t * (p2 - p1)
            self.obstacles.append({'pos': pos, 'radius': radius, 'vel': np.array([0.0,0.0])})

    def step(self, action):
        self.action_buffer.append(action)
        delayed_action = self.action_buffer[0]
        
        # --- Atualiza Obstáculos Dinâmicos ---
        if self.env_type == 'dynamic':
            for obs in self.obstacles:
                # Move
                obs['pos'] += obs.get('vel', np.array([0.0, 0.0]))
                
                # Rebate nas paredes (Ping Pong)
                if obs['pos'][0] < obs['radius'] or obs['pos'][0] > 6 - obs['radius']:
                    obs['vel'][0] *= -1
                if obs['pos'][1] < obs['radius'] or obs['pos'][1] > 6 - obs['radius']:
                    obs['vel'][1] *= -1

        # --- Física do Carro ---
        if delayed_action == 1: # FWD
            self.car_pos[0] += self.car_speed * math.cos(self.car_angle)
            self.car_pos[1] += self.car_speed * math.sin(self.car_angle)
        elif delayed_action == 2: # BACK
            self.car_pos[0] -= self.car_speed * math.cos(self.car_angle)
            self.car_pos[1] -= self.car_speed * math.sin(self.car_angle)
        elif delayed_action == 3: # LEFT
            self.car_angle -= 0.2
        elif delayed_action == 4: # RIGHT
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
        
        current_obs = np.array(self.sensors, dtype=np.float32)
        self.obs_stack.append(current_obs)
        
        # --- Rastreamento ---
        dist_moved = np.linalg.norm(self.car_pos - self.last_pos)
        if dist_moved < 0.01:
            self.steps_idle += 1
        self.last_pos = self.car_pos.copy()
        
    

        if collided:
            self.collision_count = 1

        # --- Recompensa (Lógica original preservada) ---
        d_left, d_center, d_right = self.sensors
        is_blocked_front = d_center < self.threshold_blocked_front
        is_in_corner = (d_left < self.threshold_corner) and (d_right < self.threshold_corner)

        reward = 0
        terminated = False
        
        
        if collided:
            reward = self.rewards_config.get('collision', -1.0)
            terminated = True
        else:
            if is_in_corner:
                corner_rewards = self.rewards_config.get('states',{}).get('in_corner', {})
                if delayed_action == ACTIONS_INDEX['action_back']: 
                    reward += corner_rewards.get('action_back', 0.5) 
                else: 
                    reward += corner_rewards.get('default', -0.8) 
            elif is_blocked_front:
                blocked_rewards = self.rewards_config.get('states',{}).get('blocked_front', {})
                if delayed_action == ACTIONS_INDEX['action_back']: 
                    reward += blocked_rewards.get('action_back', 0.2)
                elif delayed_action in ACTIONS_INDEX['action_turn']: 
                    reward += blocked_rewards.get('action_turn', 0.1)
                elif delayed_action == ACTIONS_INDEX['action_fwd']: 
                    reward += blocked_rewards.get('action_fwd', -1.0) 
                else: 
                    reward += blocked_rewards.get('default', -1.0) 
            else:
                clear_path_rewards = self.rewards_config.get('states',{}).get('clear_path', {})
                if delayed_action == ACTIONS_INDEX['action_fwd']: 
                    reward += clear_path_rewards.get('action_fwd', 1.0)
                elif delayed_action == ACTIONS_INDEX['action_stop']: 
                    reward += clear_path_rewards.get('action_stop', -2.0)
                elif delayed_action == ACTIONS_INDEX['action_back']: 
                    reward += clear_path_rewards.get('action_back', -5.0)
                elif delayed_action in ACTIONS_INDEX['action_turn']: 
                    reward += clear_path_rewards.get('action_turn', -0.5)
                else:
                    reward += clear_path_rewards.get('default', -1.0)
            
            penalities = self.rewards_config.get('penalities', {})
            # Penalidade por indecisão (Ações opostas consecutivas)
            if (self.last_action == 3 and delayed_action == 4) or\
                (self.last_action == 4 and delayed_action == 3) or \
                (delayed_action == 1 and self.last_action == 2) or \
                (delayed_action == 2 and self.last_action == 1):
                reward += penalities.get('indecision', -0.8)
            
            
            self.last_action = action
            reward += self.action_weights.get(delayed_action, 0)
            
            # Penalidade se não sair do lugar (Anti-Trapaça)
            if dist_moved < self.threshold_idle_movement:
                reward -= penalities.get('idle', -1.0)
            
            # Penalidade de proximidade
            min_reading = min(self.sensors)
            if min_reading < self.threshold_proximity_danger:
                reward -= (self.threshold_proximity_danger - min_reading) * penalities.get('proximity_factor', 1.0)

        if self.render_mode == "human":
            self.render()

        dist_from_start = np.linalg.norm(self.car_pos - np.array([3.0, 3.0])) # Distância da origem (não do spawn)
        
        info = {
            "is_idle": dist_moved < 0.01,
            "collision": collided,
            "dist_traveled": dist_from_start
        }
        
        
        if self.steps_idle > 50:
            terminated = True
            reward -= 50.0 # Punição nuclear
            info['collision'] = True # Trata como se fosse um acidente grave
        
        
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        return np.concatenate(self.obs_stack)

    def _update_sensors(self):
        angles = [self.car_angle - 0.78, self.car_angle, self.car_angle + 0.78]
        max_range = 3.0
        
        self.sensors = []
        for angle in angles:
            dist = max_range
            ray_dir = np.array([math.cos(angle), math.sin(angle)])
            
            # Paredes
            if ray_dir[0] != 0:
                d1 = (0 - self.car_pos[0]) / ray_dir[0] if ray_dir[0] < 0 else (6 - self.car_pos[0]) / ray_dir[0]
                if 0 < d1 < dist: dist = d1
            if ray_dir[1] != 0:
                d2 = (0 - self.car_pos[1]) / ray_dir[1] if ray_dir[1] < 0 else (6 - self.car_pos[1]) / ray_dir[1]
                if 0 < d2 < dist: dist = d2
                
            # Obstáculos (Círculos)
            for obs in self.obstacles:
                v = obs['pos'] - self.car_pos
                proj = np.dot(v, ray_dir)
                if proj > 0:
                    closest_point = self.car_pos + proj * ray_dir
                    dist_to_center = np.linalg.norm(closest_point - obs['pos'])
                    if dist_to_center < obs['radius']:
                        back_dist = math.sqrt(obs['radius']**2 - dist_to_center**2)
                        hit_dist = proj - back_dist
                        if 0 < hit_dist < dist:
                            dist = hit_dist

            self.sensors.append(dist / max_range)

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size+200, self.window_size))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size+600, self.window_size))
        canvas.fill((255, 255, 255))
        
        # Desenha Obstáculos
        for obs in self.obstacles:
            pos_px = (obs['pos'] * self.map_scale).astype(int)
            rad_px = int(obs['radius'] * self.map_scale)
            
            color = (100, 100, 100)
            if self.env_type == 'dynamic': color = (150, 0, 0) # Vermelho para dinâmicos
            if self.env_type == 'maze': color = (50, 50, 150) # Azul para labirinto
            
            pygame.draw.circle(canvas, color, pos_px, rad_px)
            
        # Desenha Carro
        car_px = (self.car_pos * self.map_scale).astype(int)
        # Corpo
        pygame.draw.circle(canvas, (0, 0, 255), car_px, int(self.car_radius * self.map_scale))
        # Indicador de direção (Linha preta)
        dir_x = car_px[0] + int(math.cos(self.car_angle) * 20)
        dir_y = car_px[1] + int(math.sin(self.car_angle) * 20)
        pygame.draw.line(canvas, (0,0,0), car_px, (dir_x, dir_y), 3)
        
        # Desenha Sensores
        angles = [self.car_angle - 0.78, self.car_angle, self.car_angle + 0.78]
        for i, dist_norm in enumerate(self.sensors):
            dist_m = dist_norm * 3.0
            end_x = self.car_pos[0] + dist_m * math.cos(angles[i])
            end_y = self.car_pos[1] + dist_m * math.sin(angles[i])
            end_px = (np.array([end_x, end_y]) * self.map_scale).astype(int)
            
            color = (255, 0, 0) if dist_norm < 0.2 else (0, 255, 0)
            pygame.draw.line(canvas, color, car_px, end_px, 2)
            
            
        # Área lateral direita (600 a 800)
        debug_x = 610
        debug_y = 50
        cell_w = 50
        cell_h = 30
        
        # Desenha título
        font = pygame.font.SysFont(None, 24)
        img = font.render("Sensor Stack (T0..T-3)", True, (0,0,0))
        canvas.blit(img, (debug_x, 10))
        
        # Desenha matriz
        # Stack é uma deque. O último elemento é o mais recente.
        # Vamos desenhar de cima pra baixo: T (recente) -> T-3 (antigo)
        stack_list = list(self.obs_stack)
        stack_list.reverse() # Mais recente em cima
        
        sensor_names = ["E", "C", "D"]
        
        for row_idx, sensors in enumerate(stack_list):
            for col_idx, val in enumerate(sensors): # val: 0.0 (perto) a 1.0 (longe)
                x = debug_x + col_idx * cell_w
                y = debug_y + row_idx * cell_h
                
                # Cor baseada no valor (Vermelho=Perto, Verde=Longe)
                # 0.0 -> (255, 0, 0)
                # 1.0 -> (0, 255, 0)
                r = int((1.0 - val) * 255)
                g = int(val * 255)
                color = (r, g, 0)
                
                pygame.draw.rect(canvas, color, (x, y, cell_w-2, cell_h-2))
                
                # Valor texto
                txt = font.render(f"{val:.1f}", True, (0,0,0))
                canvas.blit(txt, (x+10, y+5))
                
                if row_idx == 0: # Cabeçalho
                    head = font.render(sensor_names[col_idx], True, (0,0,0))
                    canvas.blit(head, (x+15, debug_y - 20))

        # Desenha Ação Atrasada (Buffer)
        buffer_y = debug_y + 300
        img = font.render(f"Action Delay ({self.latency_steps} steps):", True, (0,0,0))
        canvas.blit(img, (debug_x, buffer_y))
        
        buffer_list = list(self.action_buffer)
        for i, act in enumerate(buffer_list):
            # Desenha fila
            pygame.draw.rect(canvas, (200, 200, 200), (debug_x + i*20, buffer_y + 30, 18, 18))
            txt = font.render(str(act), True, (0,0,0))
            canvas.blit(txt, (debug_x + i*20 + 5, buffer_y + 32))


        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            