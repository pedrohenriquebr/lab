import os
import time
import numpy as np
import torch
import argparse
import subprocess
import mlflow
from datetime import datetime, timedelta
from contextlib import nullcontext # <--- O SALVADOR DA PÁTRIA

from esp32robot.config import COLAB_MODE, MODELS_DIR
from esp32robot.simulation import Esp322DEnv
from esp32robot.agents import QAgent2D
from IPython.display import clear_output

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
        if not self.episode_times: return "Calculando..."
        
        avg = sum(self.episode_times) / len(self.episode_times)
        left = self.total_episodes - (current_episode + 1)
        est = left * avg
        
        if est < 60: time_str = f"{est:.0f}s"
        elif est < 3600: time_str = f"{est/60:.1f}min"
        else: time_str = f"{est/3600:.1f}h"
        
        finish = datetime.now() + timedelta(seconds=est)
        return f"⏱️ ETA: {time_str} (~{finish.strftime('%H:%M')}) | Ep {current_episode+1}/{self.total_episodes}"

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode('utf-8')
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('utf-8')
        return commit, branch
    except:
        return "unknown", "unknown"

def train_agent(model_name='q_table_pc', episodes=50, max_steps=30, batch_size=32, display_every=5, 
                learning_rate=0.0005,
                epsilon_decay=0.98):
    
    # Configura experimento
    
    # --- LÓGICA DE CONTEXTO ---
    # Se já existe uma Run ativa (Optuna), usamos nullcontext (não faz nada)
    # Se não existe, usamos mlflow.start_run (cria nova)
    active_run = mlflow.active_run()
    if active_run:
        print(f"🔄 Anexando à Run existente: {active_run.info.run_id}")
        run_context = nullcontext()
    else:
        mlflow.set_experiment("ESP32_Robot_Navigation_2D")
        run_name = f"{model_name}_{int(time.time())}"
        print(f"✨ Criando nova Run: {run_name}")
        run_context = mlflow.start_run(run_name=run_name)

    # Inicializa variáveis fora do try para garantir acesso no finally
    agent = None
    env = None
    reward_history = []
    
    # Inicia o bloco MLflow (seja ele novo ou existente)
    with run_context:
        try:
            # Logs de Ambiente
            git_commit, git_branch = get_git_info()
            mlflow.set_tag("git.commit", git_commit)
            mlflow.set_tag("git.branch", git_branch)
            mlflow.set_tag("user", "Pedro")
            
            # Inicialização
            env = Esp322DEnv(render_mode="human")
            
            # Nota: Certifique-se que seu QAgent2D aceita 'lr' e 'epsilon_decay' no __init__
            # Se não aceitar, definimos manualmente abaixo
            agent = QAgent2D(env.action_space, use_dqn=True, batch_size=batch_size)
            agent.lr = learning_rate          # Força atualização
            agent.optimizer.param_groups[0]['lr'] = learning_rate # Atualiza otimizador
            agent.epsilon_decay = epsilon_decay # Força atualização
            
            model_file_path = str(MODELS_DIR / f"{model_name}.pth")

            # Log de Parâmetros
            mlflow.log_params({
                "episodes": episodes,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "epsilon_decay": epsilon_decay,
                "learning_rate": learning_rate,
                "architecture": "DQN_3_64_64_5"
            })

            # Carregar existente?
            if os.path.exists(model_file_path):
                agent.load(model_file_path) # Seu load adiciona .pth
                if hasattr(agent, 'target_net'):
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
                print("✅ Modelo carregado!")

            timer = TrainingTimer(total_episodes=episodes, display_every=display_every)
            
            print(f"\n🚀 Treino Iniciado: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

            # Loop Principal
            for ep in range(episodes):
                timer.start_episode()
                obs, _ = env.reset()
                total_reward = 0
                episode_losses = []

                for step in range(max_steps):
                    action = agent.get_action(obs)
                    next_obs, reward, done, _, _ = env.step(action)
                    
                    # time.sleep(0.01) # Pequeno delay para visualização se necessário
                    
                    loss = agent.update(obs, action, reward, next_obs)
                    if loss and loss != 0:
                        episode_losses.append(loss)
                        # Log detalhado (cuidado com tamanho do log)
                        # mlflow.log_metric("step_loss", loss, step=ep*max_steps+step)

                    obs = next_obs
                    total_reward += reward
                    if done: break

                reward_history.append(total_reward)

                if agent.epsilon > agent.min_epsilon:
                    agent.epsilon *= agent.epsilon_decay

                avg_loss = np.mean(episode_losses) if episode_losses else 0
                
                # Log de Métricas por Episódio
                mlflow.log_metrics({
                    "reward": total_reward,
                    "avg_loss": avg_loss,
                    "epsilon": agent.epsilon
                }, step=ep)

                if timer.end_episode(ep):
                    if COLAB_MODE: clear_output(wait=True)
                    print(f"{timer.get_estimate(ep)} | R: {total_reward:6.1f} | L: {avg_loss:6.4f} | Eps: {agent.epsilon:.2f}")
                    agent.plot_training_results()

            print("\n✅ Treino concluído!")
            
            # Salvar Artefatos Finais
            agent.save(model_file_path) # Seu save adiciona .pth
            mlflow.log_artifact(model_file_path)
            
            return agent, env, reward_history

        except KeyboardInterrupt:
            print("\n⛔ Interrompido pelo usuário!")
            if agent:
                agent.save(model_file_path)
                print("💾 Modelo salvo antes de sair.")
            return agent, env, reward_history
            
        finally:
            if env: env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="model_default", help="Nome do experimento/modelo")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()

    # Monta o caminho completo
    model_save_name = f"{args.name}.pth" # O path completo será tratado na função save

    print(f"🧪 Iniciando Experimento: {args.name}")

    AGENTE_TREINADO, AMBIENTE, HISTORICO = train_agent(
        model_name=args.name, # Passa o nome limpo
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        display_every=1
    )

    print("\n📊 Resultado Final:")
    print(f"Média de Recompensa Final: {np.mean(HISTORICO[-10:]):.1f}")
    print("Modelo salvo em: q_table_pc.pth")
    print("Gráfico salvo em: loss_chart.png")
    print('Historico de recompensas: ')
    print(HISTORICO[:50])