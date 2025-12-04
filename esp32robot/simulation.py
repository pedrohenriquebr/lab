import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random

class Esp322DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, env_type='default'):
        self.window_size = 600  # Tamanho da janela (pixels)
        self.map_scale = 100    # 100 pixels = 1 metro (Mundo de 6x6 metros)
        
        # Tipos: 'default', 'maze', 'circular_race_track', 'dynamic'
        self.env_type = env_type 
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Action Space: 0:Stop, 1:Fwd, 2:Back, 3:Left, 4:Right
        self.action_space = spaces.Discrete(5)
        
        # Observation Space: 3 valores flutuantes (Distância Esq, Centro, Dir)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.car_radius = 0.15 
        self.car_speed = 0.05
        
        # Estado Inicial
        self.car_pos = np.array([3.0, 3.0])
        self.car_angle = 0.0
        
        self.obstacles = []
        self.sensors = [0.0, 0.0, 0.0]

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
        
        self._update_sensors()
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
        
        # --- Rastreamento ---
        dist_moved = np.linalg.norm(self.car_pos - self.last_pos)
        if dist_moved < 0.01:
            self.steps_idle += 1
        self.last_pos = self.car_pos.copy()

        if collided:
            self.collision_count = 1

        # --- Recompensa (Lógica original preservada) ---
        d_left, d_center, d_right = self.sensors
        is_blocked_front = d_center < 0.25 
        is_in_corner = (d_left < 0.25) and (d_right < 0.25)

        reward = 0
        terminated = False

        if collided:
            reward = -1.0
            terminated = True
        else:
            if is_in_corner:
                if action == 2: reward += 0.5 
                else: reward -= 0.8 
            elif is_blocked_front:
                if action == 2: reward += 0.2
                elif action in [3, 4]: reward += 0.1 
                elif action == 1: reward -= 1.0
                else: reward -= 0.1
            else:
                if action == 1: reward += 0.2
                elif action == 0: reward -= 0.05
                elif action in [3, 4]: reward -= 0.05
                elif action == 2: reward -= 0.1

            # Penalidade se não sair do lugar (Anti-Trapaça)
            if dist_moved < 0.01:
                reward -= 0.1
            
            # Penalidade de proximidade
            min_reading = min(self.sensors)
            if min_reading < 0.3:
                reward -= (0.3 - min_reading) * 1.0

        if self.render_mode == "human":
            self.render()

        dist_from_start = np.linalg.norm(self.car_pos - np.array([3.0, 3.0])) # Distância da origem (não do spawn)
        
        info = {
            "is_idle": dist_moved < 0.01,
            "collision": collided,
            "dist_traveled": dist_from_start
        }
        
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        return np.array(self.sensors, dtype=np.float32)

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
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
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

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()