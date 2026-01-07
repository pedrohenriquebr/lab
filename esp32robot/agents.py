from collections import defaultdict
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
from gymnasium import spaces
from gymnasium import core
import torch.nn as nn
import torch.nn.functional as F


class DiscreteWorldModel(nn.Module):
    def __init__(self, input_dim=12, n_actions=5, n_categorias=32, hidden_size=64):
        super().__init__()
        self.n_categorias = n_categorias
        self.n_actions = n_actions
        # Mesma arquitetura do seu treino (BatchNorm + 64 cats)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_categorias)
        )
        # O resto (Decoder/Predictor) não precisamos carregar na RAM do agente 
        # para economizar, mas a classe precisa ter a estrutura pra carregar os pesos
        self.decoder = nn.Sequential(
            nn.Linear(n_categorias, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, input_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(n_categorias + n_actions, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_categorias)
        )

    def forward(self, x, action=None, hard=False):
        # 1. ENCODER: Gera logits e usa Gumbel-Softmax para discretizar
        logits = self.encoder(x)
        # hard=True retorna one-hot (ex: [0, 1, 0]) mas permite gradiente passar
        z_dist = F.gumbel_softmax(logits, tau=1.0, hard=hard, dim=1)

        # 2. DECODER: Tenta reconstruir a entrada original
        reconstrucao = self.decoder(z_dist)

        # 3. PREDICTOR: Se tivermos ação, tenta prever o futuro
        z_future_logits = None
        if action is not None:
            # One-hot da ação
            action_onehot = F.one_hot(action.long(), num_classes=self.n_actions).float()
            
            # Concatena estado latente atual + ação
            pred_input = torch.cat([z_dist, action_onehot], dim=1)
            z_future_logits = self.predictor(pred_input)

        return reconstrucao, z_dist, z_future_logits, logits

    def get_latent_index(self, x):
        """Método utilitário para saber qual 'Número' o modelo escolheu"""
        logits = self.encoder(x)
        return torch.argmax(logits, dim=1)

    def get_category_probs(self, x):
        """Retorna as probabilidades (Softmax) ou One-Hot"""
        self.eval()
        with torch.no_grad():
            logits = self.encoder(x)
            # Retorna probabilidades reais (útil para o DQN)
            return F.softmax(logits, dim=1)

    def get_category_index(self, x):
        """Retorna apenas o ID (útil para Q-Table ou Debug)"""
        probs = self.get_category_probs(x)
        return torch.argmax(probs, dim=1).item()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VisualWorldModel(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), n_actions=5, n_categorias=32, hidden_size=128):
        """
        input_shape: Tupla (Canais, Altura, Largura). Ex: (3, 64, 64) ou (3, 96, 96)
        """
        super().__init__()
        self.n_categorias = n_categorias
        self.n_actions = n_actions
        
        c, h, w = input_shape # Desempacota (3, 64, 64)
        
        # --- 1. ENCODER (FLEXÍVEL) ---
        self.encoder_cnn = nn.Sequential(
            # Camada 1: Divide por 2
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Camada 2: Divide por 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Camada 3: Divide por 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Camada 4: Divide por 2
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # --- CÁLCULO AUTOMÁTICO DO TAMANHO ---
        # Criamos um tensor falso com o tamanho da entrada para ver o que sai
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            dummy_output = self.encoder_cnn(dummy_input)
            
            # Pega o formato de saída (Ex: [1, 256, 4, 4])
            self.feature_shape = dummy_output.shape[1:] # Ignora o batch: (256, 4, 4)
            self.flatten_size = dummy_output.view(1, -1).size(1) # Total de neurônios: 4096
            
            print(f"📐 Auto-Config: Input {h}x{w} -> CNN Sai {self.feature_shape[1]}x{self.feature_shape[2]} -> Linear {self.flatten_size}")

        # Agora criamos o Linear com o tamanho exato calculado
        self.encoder_head = nn.Linear(self.flatten_size, n_categorias)

        # --- 2. DECODER (FLEXÍVEL) ---
        # Faz o caminho reverso exato
        self.decoder_head = nn.Linear(n_categorias, self.flatten_size)
        
        self.decoder_cnn = nn.Sequential(
            # Unflatten manual no forward usando self.feature_shape
            
            # Deconv 1: x2
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Deconv 2: x2
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Deconv 3: x2
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Deconv 4: x2
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # --- 3. PREDICTOR (Igual) ---
        self.predictor = nn.Sequential(
            nn.Linear(n_categorias + n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_categorias)
        )
        
        
        self.inverse_head = nn.Sequential(
            nn.Linear(n_categorias * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x, action=None, hard=False, temperature=1.0):
        batch_size = x.size(0)
        
        # --- Encoder ---
        features = self.encoder_cnn(x)
        # Usa reshape dinâmico
        features = features.reshape(batch_size, -1) 
        logits = self.encoder_head(features)
        
        z_dist = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=1)

        # --- Decoder ---
        z_dec = self.decoder_head(z_dist)
        # Unflatten dinâmico usando o shape calculado no __init__
        # self.feature_shape é (256, H_out, W_out)
        z_dec = z_dec.reshape(batch_size, *self.feature_shape) 
        reconstrucao = self.decoder_cnn(z_dec)

        # --- Predictor ---
        z_future_logits = None
        if action is not None:
            if isinstance(action, torch.Tensor) and action.dim() == 1:
                action = action.long()
                action_onehot = F.one_hot(action, num_classes=self.n_actions).float()
            elif isinstance(action, torch.Tensor) and action.dim() == 2:
                action_onehot = action.float()
            else:
                action_onehot = F.one_hot(torch.tensor(action), num_classes=self.n_actions).float().to(x.device)

            pred_input = torch.cat([z_dist, action_onehot], dim=1)
            z_future_logits = self.predictor(pred_input)

        return reconstrucao, z_dist, z_future_logits, logits
    
    def get_latent_probs(self, x):
        self.eval()
        with torch.no_grad():
            feat = self.encoder_cnn(x)
            feat = feat.reshape(x.size(0), -1)
            logits = self.encoder_head(feat)
            return F.softmax(logits, dim=1)

    def predict_action_inverse(self, z_current, z_next):
        """
        Tenta adivinhar qual ação levou de z_current para z_next.
        """
        # Concatena os dois vetores latentes
        combined = torch.cat([z_current, z_next], dim=1)
        action_logits = self.inverse_head(combined)
        return action_logits

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)



class QAgent2D:
    def __init__(self, action_space: spaces.Space[core.ActType],
                    observation_space: spaces.Space[core.ObsType], 
                    use_dqn=False, batch_size=64,
                    gamma=0.9, 
                    learning_rate=0.0005,
                    world_model_path="mini_world_model.pth",
                    epsilon_start=1.0,
                    min_epsilon=0.05,
                    epsilon_decay=0.995,
                    target_update_freq=100,
                    memory_size=5000,
                    world_model_n_categories=32,
                    world_model_hidden_size=64,
                    debug=True): # <--- CAMINHO DO MODELO
            
            self.q_table = defaultdict(lambda: np.zeros(action_space.n)) 
            self.lr = learning_rate
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_decay = epsilon_decay
            self.min_epsilon = min_epsilon
            self.action_space = action_space
            self.loss_history: list[float] = []
            self.reward_history: list[float] = []
            self.target_update_freq = target_update_freq
            self.step_count = 0
            self.use_dqn = use_dqn
            self.previous_fig_and_ax = None
            self.memory_size = memory_size
            
            # --- CARREGAR WORLD MODEL ---
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Instancia o World Model (64 categorias)
            if debug:
                print("🧠 Inicializando World Model...")
                print(f"   Dispositivo: {self.device}")
                print(f"   Espaço de Ação: {action_space}")
                print(f"   Espaço de Observação: {observation_space}")
                
            self.wm = DiscreteWorldModel(input_dim=observation_space.shape[0], 
                                        n_categorias=world_model_n_categories,
                                        hidden_size=world_model_hidden_size).to(self.device)
            
            self.n_categorias = world_model_n_categories
            
            if os.path.exists(world_model_path):
                print(f"🧠 Carregando World Model de: {world_model_path}")
                # Carrega pesos (ignore erros de keys faltantes se o decoder não bater exato, 
                # mas idealmente deve bater a arquitetura)
                try:
                    self.wm.load_state_dict(torch.load(world_model_path, map_location=self.device))
                except:
                    print("⚠️ Aviso: Pesos carregados com 'strict=False' (Pode ser normal se mudou algo)")
                    self.wm.load_state_dict(torch.load(world_model_path, map_location=self.device), strict=False)
                self.wm.eval() # O WM não treina mais, ele só "enxerga"
            else:
                print(f"❌ ERRO CRÍTICO: World Model não encontrado em {world_model_path}!")
                print("O agente vai rodar cego (aleatório) se não arrumar isso.")

            # --- CONFIGURA O DQN PARA O NOVO ESPAÇO ---
            if self.use_dqn:
                print(f"🤖 Agente DDQN no dispositivo: {self.device}")
                # O input do DQN agora é o tamanho do One-Hot Vector (64)
                dqn_input_dim = self.n_categorias 
                
                self.policy_net = DQN(dqn_input_dim, action_space.n).to(self.device)
                self.target_net = DQN(dqn_input_dim, action_space.n).to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.eval()

                self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
                self.loss_fn = nn.MSELoss()
                self.memory : list[tuple] = []
                self.batch_size = batch_size
                self.optimizer.param_groups[0]['lr'] = learning_rate # Atualiza otimizador
                
    def get_arch(self) -> str:
                in_dim = self.policy_net.fc1.in_features
                h1_dim = self.policy_net.fc1.out_features
                return f'DQN_WM_{in_dim}_{h1_dim}'

    def _default_discretize_sensors(self, sensors):
        """
        Converte os 3 valores contínuos (0.0 a 1.0) em estados discretos (0, 1, 2).
        Isso simula o que o VisionProcessor fará no robô real.
        
        Input: [esq_dist, centro_dist, dir_dist] (Normalizado: 0=Colado, 1=Livre)
        
        Lógica Inversa do ambiente (lá 0=colado):
        Se dist < 0.15 (Muito Perto) -> 2 (Perigo/Colisão iminente)
        Se dist < 0.40 (Perto)       -> 1 (Atenção)
        Se dist >= 0.40 (Longe)      -> 0 (Livre)
        """
        state = []
        for d in sensors:
            # Nota: No ambiente 2D, o sensor retorna a distância normalizada (0 a 1)
            # onde 1.0 é longe (3 metros) e 0.0 é colado.
            
            if d < 0.15:   # Menos de 45cm
                val = 2    # Perigo
            elif d < 0.40: # Menos de 1.2m
                val = 1    # Atenção
            else:
                val = 0    # Livre
            state.append(val)
            
        return tuple(state)
        
    def discretize_sensors(self, sensors):
            """
            A MÁGICA: Converte sensores (12 floats) -> Conceito (One-Hot ou Int)
            """
            # Converte para Tensor
            sensors_t = torch.tensor(sensors, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Roda o Encoder
            probs = self.wm.get_category_probs(sensors_t) # Shape [1, 64]
            
            if self.use_dqn:
                # Para DQN, retornamos o vetor de probabilidades (Soft State) ou One-Hot
                # Isso dá mais informação pro DQN do que apenas um inteiro.
                return probs.cpu().numpy()[0] # Retorna array numpy (64,)
            else:
                # Para Q-Table, precisamos de um Inteiro (Hashable)
                idx = torch.argmax(probs).item()
                return int(idx)
    
    def _discretize_sensors_dqn(self, sensors: np.ndarray):
        return tuple(sensors)
    
    # def discretize_sensors(self, sensors):
    #     if not self.use_dqn:
    #         return self._default_discretize_sensors(sensors)
    #     return self._discretize_sensors_dqn(sensors)

    def get_action(self, sensors):
            # Transforma a visão em conceito
            state = self.discretize_sensors(sensors) # Pode ser Int ou Array(64,)
            
            if np.random.random() < self.epsilon:
                return self.action_space.sample()
            
            if self.use_dqn:
                # O state já é um vetor (64,). Transformar em tensor batch (1, 64)
                state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    current_q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                selected_action = int(np.argmax(current_q_values))
            else:
                # Q-Table usa o inteiro como chave
                current_q_values = self.q_table[state]
                selected_action = int(np.argmax(current_q_values))

            return selected_action

    def update(self, sensors, action, reward, next_sensors):
        # Converte sensores brutos -> Conceitos Latentes
        state = self.discretize_sensors(sensors)
        next_state = self.discretize_sensors(next_sensors)
        
        if not self.use_dqn:
            # Lógica Q-Table (Estado é Int)
            best_next = np.max(self.q_table[next_state])
            current_q = self.q_table[state][action]
            new_q = current_q + self.lr * (reward + self.gamma * best_next - current_q)
            self.q_table[state][action] = new_q
        else:
            # Lógica DQN (Estado é Array 64,)
            self.memory.append((state, action, reward, next_state))
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)

            if len(self.memory) < self.batch_size:
                return 0

            batch = random.sample(self.memory, self.batch_size)
            
            # Aqui state já é o vetor one-hot/probs, então empilhamos direto
            states_b = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(self.device)
            actions_b = torch.tensor([[x[1]] for x in batch], dtype=torch.long).to(self.device)
            rewards_b = torch.tensor([x[2] for x in batch], dtype=torch.float32).to(self.device)
            next_states_b = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                next_actions = self.policy_net(next_states_b).max(1)[1].unsqueeze(1)
                q_next = self.target_net(next_states_b).gather(1, next_actions).squeeze()
                q_target = rewards_b + (self.gamma * q_next)

            q_curr = self.policy_net(states_b).gather(1, actions_b).squeeze()
            loss = self.loss_fn(q_curr, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(loss.item())
            self.reward_history.append(reward)
            self.step_count += 1

            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            return loss.item()

        return 0
    
    def plot_training_results(self):
        if not self.use_dqn or not self.loss_history:
            return
        
        # backend 'Agg' é essencial para não tentar abrir janelas GUI e travar o PyGame
        import matplotlib
        matplotlib.use('Agg') 
        import matplotlib.pyplot as plt

        # Cria figura nova a cada vez para evitar memory leak do Matplotlib
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))

        # ==========================================
        # GRÁFICO 1: LOSS
        # ==========================================
        ax[0].plot(self.loss_history, label='Loss (MSE)', color='red', alpha=0.4)

        if len(self.loss_history) > 50:
            window = 50
            moving_avg_loss = np.convolve(self.loss_history, np.ones(window)/window, mode='valid')
            ax[0].plot(range(window-1, len(self.loss_history)), moving_avg_loss, label='Média Móvel', color='darkred', linewidth=2)

        ax[0].set_title('Evolução do Erro (Loss)')
        ax[0].set_ylabel('Erro (MSE)')
        ax[0].legend()
        ax[0].grid(True)

        # ==========================================
        # GRÁFICO 2: RECOMPENSAS
        # ==========================================
        if hasattr(self, 'reward_history') and self.reward_history:
            ax[1].plot(self.reward_history, label='Reward (Passo a Passo)', color='green', alpha=0.3)

            if len(self.reward_history) > 50:
                window = 50
                moving_avg_rew = np.convolve(self.reward_history, np.ones(window)/window, mode='valid')
                ax[1].plot(range(window-1, len(self.reward_history)), moving_avg_rew, label='Tendência (Média)', color='blue', linewidth=2)

            ax[1].set_title('Histórico de Recompensas')
            ax[1].set_xlabel('Passos de Treinamento (Steps)')
            ax[1].set_ylabel('Recompensa')
            ax[1].legend()
            ax[1].grid(True)
        else:
            ax[1].text(0.5, 0.5, "Sem dados de reward_history", ha='center')

        fig.tight_layout()
        
        # SALVA E FECHA (Não usa plt.show)
        try:
            plt.savefig("training_charts.png")
        except:
            pass
        finally:
            plt.close(fig) # CRÍTICO: Libera a memória e a thread para o PyGame
            plt.close('all') 
        
    def save(self, filename):
        if self.use_dqn:
            torch.save(self.policy_net.state_dict(), f"{filename}")
            print(f"Modelo salvo: {filename}")
        else:
            with open(f"{filename}", 'wb') as f:
                pickle.dump(dict(self.q_table), f)
            print(f"Q-Table salva: {filename}")

    def load(self, filename):
        if self.use_dqn:
            checkpoint = torch.load(f"{filename}", map_location=self.device)
            self.policy_net.load_state_dict(checkpoint)
            print(f"Modelo carregado: {filename}")
