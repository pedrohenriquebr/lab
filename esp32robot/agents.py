from collections import defaultdict
import pickle
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
from gymnasium import spaces
from gymnasium import core

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
                 learning_rate=0.0005):
        self.q_table = defaultdict(lambda: np.zeros(action_space.n)) # type: ignore
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        self.action_space = action_space
        self.loss_history: list[float] = []
        self.reward_history: list[float] = []
        self.target_update_freq = 100
        self.step_count = 0
        self.use_dqn = use_dqn
        self.previous_fig_and_ax = None
        
        if self.use_dqn:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Agente DDQN no dispositivo: {self.device}")
            self.policy_net = DQN(observation_space.shape[0], action_space.n).to(self.device)
            self.target_net = DQN(observation_space.shape[0], action_space.n).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
            self.loss_fn = nn.MSELoss()
            self.memory : list[tuple] = []
            self.batch_size = batch_size
            
    def get_arch(self) -> str:
            # Pega dimensões reais de cada camada
            in_dim = self.policy_net.fc1.in_features
            h1_dim = self.policy_net.fc1.out_features
            h2_dim = self.policy_net.fc2.out_features
            out_dim = self.policy_net.fc3.out_features
            
            return f'DQN_{in_dim}_{h1_dim}_{h2_dim}_{out_dim}'

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
    
    def _discretize_sensors_dqn(self, sensors: np.ndarray):
        return tuple(sensors)
    
    def discretize_sensors(self, sensors):
        if not self.use_dqn:
            return self._default_discretize_sensors(sensors)
        return self._discretize_sensors_dqn(sensors)

    def get_action(self, sensors):
        state = self.discretize_sensors(sensors)
        
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        if self.use_dqn:
            state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                current_q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            selected_action = int(np.argmax(current_q_values))
        else:
            current_q_values = self.q_table[state]
            selected_action = int(np.argmax(current_q_values))

        return selected_action

    def update(self, sensors, action, reward, next_sensors):
        state = self.discretize_sensors(sensors)
        next_state = self.discretize_sensors(next_sensors)
        
        if not self.use_dqn:
            best_next = np.max(self.q_table[next_state])
            current_q = self.q_table[state][action]
            new_q = current_q + self.lr * (reward + self.gamma * best_next - current_q)
            self.q_table[state][action] = new_q
        else:
            self.memory.append((state, action, reward, next_state))
            if len(self.memory) > 2000:
                self.memory.pop(0)

            if len(self.memory) < self.batch_size:
                return 0

            batch = random.sample(self.memory, self.batch_size)
            states_b = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(self.device)
            actions_b = torch.tensor([[x[1]] for x in batch], dtype=torch.long).to(self.device)
            rewards_b = torch.tensor([x[2] for x in batch], dtype=torch.float32).to(self.device)
            next_states_b = torch.tensor([x[3] for x in batch], dtype=torch.float32).to(self.device)

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
