import os
import time
import numpy as np
import torch
from esp32robot.config import COLAB_MODE, MODELS_DIR
from esp32robot.simulation import Esp322DEnv
from esp32robot.agents import QAgent2D
from IPython.display import clear_output
from datetime import datetime, timedelta

class TrainingTimer:
    def __init__(self, total_episodes, display_every=10):
        self.total_episodes = total_episodes
        self.display_every = display_every
        self.start_time = time.time()
        self.episode_times = []
        self.window_size = 10
        self.last_display = -1

    def start_episode(self):
        self.episode_start = time.time()

    def end_episode(self, episode_num):
        duration = time.time() - self.episode_start
        self.episode_times.append(duration)
        if len(self.episode_times) > self.window_size:
            self.episode_times.pop(0)
        return self.should_display(episode_num)

    def should_display(self, episode_num):
        return episode_num % self.display_every == 0 and episode_num != self.last_display

    def get_estimate(self, current_episode):
        self.last_display = current_episode

        if not self.episode_times:
            return "Calculando velocidade..."

        avg_time_per_ep = sum(self.episode_times) / len(self.episode_times)
        episodes_left = self.total_episodes - (current_episode + 1)
        estimated_seconds = episodes_left * avg_time_per_ep

        if estimated_seconds < 60:
            time_str = f"{estimated_seconds:.0f}s"
        elif estimated_seconds < 3600:
            mins = estimated_seconds / 60
            time_str = f"{mins:.1f}min"
        else:
            hours = estimated_seconds / 3600
            time_str = f"{hours:.1f}h"

        eps_per_sec = 1.0 / avg_time_per_ep
        finish_time = datetime.now() + timedelta(seconds=estimated_seconds)

        return (
            f"⏱️  ETA: {time_str} (~{finish_time.strftime('%H:%M')}) | "
            f"Ep {current_episode+1}/{self.total_episodes} | "
            f"{eps_per_sec:.2f} ep/s | {avg_time_per_ep:.2f}s/ep"
        )


def train_agent(model_name='q_table_pc', episodes=50, max_steps=30, batch_size=32, display_every=5):
    """Função wrapper para treinar no Colab"""

    # Limpa arquivos debug antigos
    #!rm -f /content/debug_step_*.png

    # Cria ambiente e agente
    env = Esp322DEnv(render_mode="human")
    agent = QAgent2D(env.action_space, use_dqn=True, batch_size=batch_size)
    agent.epsilon_decay = 0.98    # Carrega modelo existente (se houver)
    model_file_path = f"{MODELS_DIR/model_name}.pth"

    if os.path.exists(model_file_path):
        agent.load(model_file_path)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("✅ Modelo existente carregado!")

    timer = TrainingTimer(total_episodes=episodes, display_every=display_every)
    reward_history = []

    try:
        print("\n" + "="*60)
        print("🚀 TREINAMENTO INICIADO NO COLAB")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"Meta: {episodes} episódios, {max_steps} steps/época")
        print("="*60 + "\n")



        for ep in range(episodes):
            timer.start_episode()
            obs, _ = env.reset()
            total_reward = 0
            episode_losses = []

            for step in range(max_steps):
                action = agent.get_action(obs)
                next_obs, reward, done, _, _ = env.step(action)
                time.sleep(0.03)
                loss = agent.update(obs, action, reward, next_obs)
                if loss is not None and loss != 0:
                    episode_losses.append(loss)

                obs = next_obs
                total_reward += reward

                if done:
                    break

            reward_history.append(total_reward)

            # Decaimento do Epsilon
            if agent.epsilon > agent.min_epsilon:
                agent.epsilon *= agent.epsilon_decay

            avg_loss = np.mean(episode_losses) if episode_losses else 0

            # Display no Colab
            if timer.end_episode(ep):
                if COLAB_MODE:
                    clear_output(wait=True)
                
                print(timer.get_estimate(ep))
                print(f"   └─ Reward: {total_reward:>6.1f} | Loss: {avg_loss:>6.3f} | Epsilon: {agent.epsilon:.3f}")
                print(f"   └─ Últimos 10 Rewards: {np.mean(reward_history[-10:]):.1f} (média)")
                agent.plot_training_results()


        print("\n✅ Treinamento concluído!")
        return agent, env, reward_history

    except KeyboardInterrupt:
        print("\n⛔ Treinamento interrompido!")
        return agent, env, reward_history
    finally:
        # Salva modelo final
        agent.save(model_file_path)
        agent.plot_training_results()
        env.close()


if __name__ == "__main__":
    AGENTE_TREINADO, AMBIENTE, HISTORICO = train_agent(
        model_name='q_table_pc_2',
        episodes=50,
        max_steps=200,
        batch_size=128,
        display_every=1
    )

    print("\n📊 Resultado Final:")
    print(f"Média de Recompensa Final: {np.mean(HISTORICO[-10:]):.1f}")
    print("Modelo salvo em: q_table_pc.pth")
    print("Gráfico salvo em: loss_chart.png")
    print('Historico de recompensas: ')
    print(HISTORICO[:50])