import os
import time
import numpy as np
import torch
import argparse
import mlflow
from datetime import datetime, timedelta
from contextlib import nullcontext # <--- O SALVADOR DA PÁTRIA

from esp32robot.config import COLAB_MODE, MODELS_DIR
from esp32robot.simulation import Esp322DEnv
from esp32robot.agents import QAgent2D
from IPython.display import clear_output

from esp32robot.utils import get_git_info, load_config

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

class TrainingArguments:
    def __init__(self, environment_config: dict, agent_config: dict, world_model_config: dict, training_config: dict):
        self.environment_config = environment_config
        self.agent_config = agent_config
        self.world_model_config = world_model_config
        self.training_config = training_config


def train_agent(model_name='q_table_pc', episodes=50, max_steps=30, batch_size=32, display_every=5, 
                learning_rate=0.0005,
                epsilon_decay=0.99,
                headless=False,
                gamma=0.9,
                stack_size=4,
                latency_steps=5,
                rotation_penalty=0.1,
                experiment_name:str = 'ESP32_Robot_Navigation_2D',
                env_type:str = 'maze' ,
                use_dqn:bool = True,
                epsilon_start: float = 1.0,
                min_epsilon: float = 0.05,
                debug: bool = False,
                output_dir: str  = '',
                memory_size: int = 10000,
                target_update_freq: int = 100,
                threshold_blocked_front:float=0,
                threshold_corner:float=0,
                threshold_proximity_danger:float=0,
                threshold_idle_movement:float=0,
                rewards=None,
                world_model_path: str = 'mini_world_model',
                world_model_n_categories: int = 32,
                world_model_hidden_size: int = 64,
                ):
    
    # Configura experimento
    
    # --- LÓGICA DE CONTEXTO ---
    # Se já existe uma Run ativa (Optuna), usamos nullcontext (não faz nada)
    # Se não existe, usamos mlflow.start_run (cria nova)
    active_run = mlflow.active_run()
    if active_run:
        print(f"🔄 Anexando à Run existente: {active_run.info.run_id}")
        run_context = nullcontext()
    else:
        mlflow.set_experiment(experiment_name)
        run_name = f"{model_name}_{int(time.time())}"
        print(f"✨ Criando nova Run: {run_name}")
        run_context = mlflow.start_run(run_name=run_name)

    # Inicializa variáveis fora do try para garantir acesso no finally
    agent = None
    env = None
    reward_history = []
    crash_rate_history = []
    idle_rate_history = []
    avg_distance_history = []
    
    # Inicia o bloco MLflow (seja ele novo ou existente)
    with run_context:
        try:
            # Logs de Ambiente
            git_commit, git_branch = get_git_info()
            mlflow.set_tag("git.commit", git_commit)
            mlflow.set_tag("git.branch", git_branch)
            mlflow.set_tag("user", "Pedro")
            render_mode = "human" if not headless else None
            
            # Inicialização
            env = Esp322DEnv(render_mode=render_mode, env_type=env_type, 
                             rotation_penalty=rotation_penalty, 
                             latency_steps=latency_steps, 
                             stack_size=stack_size,
                             threshold_blocked_front=threshold_blocked_front,
                             threshold_corner=threshold_corner,
                             threshold_proximity_danger=threshold_proximity_danger,
                             threshold_idle_movement=threshold_idle_movement,
                             rewards=rewards
                             )

            # Nota: Certifique-se que seu QAgent2D aceita 'lr' e 'epsilon_decay' no __init__
            # Se não aceitar, definimos manualmente abaixo
            agent = QAgent2D(env.action_space, env.observation_space, 
                             use_dqn=use_dqn, 
                             batch_size=batch_size,
                             gamma=gamma,
                             learning_rate=learning_rate,
                             epsilon_start=epsilon_start,
                             min_epsilon=min_epsilon,
                             epsilon_decay=epsilon_decay,
                             target_update_freq=target_update_freq,
                             memory_size=memory_size,
                             world_model_path=world_model_path,
                            world_model_n_categories=world_model_n_categories,
                            world_model_hidden_size=world_model_hidden_size,
                             debug=debug)
            
            
            if output_dir is None or output_dir == '':
                model_file_path = str(MODELS_DIR / f"{model_name}.pth")
            else:
                model_file_path = str(os.path.join(os.path.join(os.getcwd(), output_dir), f"{model_name}.pth"))

            # Log de Parâmetros
            mlflow.log_params({
                "episodes": episodes,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "epsilon_decay": epsilon_decay,
                "learning_rate": learning_rate,
                "architecture": agent.get_arch()
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
                total_idle_steps = 0
                did_crash = 0
                max_dist = 0
                dist_history=[]
                crash_history=[]
                action_counts = {0:0, 1:0, 2:0, 3:0, 4:0}


                for step in range(max_steps):
                    action = agent.get_action(obs)
                    next_obs, reward, done, _, info = env.step(action)
                    action_counts[action] += 1
                    # if not headless:
                    #     time.sleep(5)  # Pequeno delay para visualização
                    
                    loss = agent.update(obs, action, reward, next_obs)
                    if loss and loss != 0:
                        episode_losses.append(loss)
                        # Log detalhado (cuidado com tamanho do log)
                        # mlflow.log_metric("step_loss", loss, step=ep*max_steps+step)

                    obs = next_obs
                    total_reward += reward
                    total_idle_steps += 1 if info['is_idle'] else 0
                    did_crash = 1 if info['collision'] else 0
                    max_dist = max(max_dist, info['dist_traveled'])
                    dist_history.append(max_dist)
                    crash_history.append(did_crash)
                    
                    
                    if done: break

                
                
                
                if agent.epsilon > agent.min_epsilon:
                    agent.epsilon *= agent.epsilon_decay

                avg_loss = np.mean(episode_losses) if episode_losses else 0
                avg_distance = np.mean(dist_history) if dist_history else 0
                idle_ratio = total_idle_steps / max_steps
                crash_rate = sum(crash_history) / max_steps
                total_actions = sum(action_counts.values())
                
                
                avg_distance_history.append(avg_distance)
                idle_rate_history.append(idle_ratio)
                crash_rate_history.append(crash_rate)
                reward_history.append(total_reward)
                
                # Log de Métricas por Episódio
                mlflow.log_metrics({
                    "reward": total_reward,
                    "avg_loss": avg_loss,
                    "epsilon": agent.epsilon,
                    "idle_ratio": idle_ratio,
                    "crash_rate": crash_rate,
                    "avg_distance": avg_distance
                }, step=ep)
                
                
                if total_actions > 0:
                    mlflow.log_metrics({
                        "act_stop_pct": action_counts[0] / total_actions,
                        "act_fwd_pct": action_counts[1] / total_actions,
                        "act_back_pct": action_counts[2] / total_actions,
                        "act_left_pct": action_counts[3] / total_actions,
                        "act_right_pct": action_counts[4] / total_actions
                    }, step=ep)

                if timer.end_episode(ep):
                    if COLAB_MODE: clear_output(wait=True)
                    print(f"{timer.get_estimate(ep)} | R: {total_reward:6.1f} | L: {avg_loss:6.4f} | Eps: {agent.epsilon:.2f}")
                    agent.plot_training_results()

            print("\n✅ Treino concluído!")
            
            metrics = {
                'reward_history': reward_history,
                'crash_rate_history': crash_rate_history,
                'idle_rate_history': idle_rate_history,
                'avg_distance_history': avg_distance_history,
            }
            
            # Salvar Artefatos Finais
            agent.save(model_file_path) # Seu save adiciona .pth
            mlflow.log_artifact(model_file_path)
            
            return agent, env, metrics

        except KeyboardInterrupt:
            print("\n⛔ Interrompido pelo usuário!")
            if agent:
                agent.save(model_file_path)
                print("💾 Modelo salvo antes de sair.")
            
            metrics = {
                'reward_history': reward_history,
                'crash_rate_history': crash_rate_history,
                'idle_rate_history': idle_rate_history,
                'avg_distance_history': avg_distance_history,
            }
            
            return agent, env, metrics
            
        finally:
            if env: env.close()





def train_agent_from_config(args: TrainingArguments):
    return train_agent(
        # environment parameters
        env_type=args.environment_config.get('env_type', 'maze'),
        stack_size=args.environment_config.get('stack_size', 4),
        latency_steps=args.environment_config.get('latency_steps', 5),
        rotation_penalty=args.environment_config.get('rotation_penalty', 0.1),
        threshold_blocked_front=args.environment_config.get('thresholds', {}).get('blocked_front', 0),
        threshold_corner=args.environment_config.get('thresholds', {}).get('corner', 0),
        threshold_proximity_danger=args.environment_config.get('thresholds', {}).get('proximity_danger', 0),
        threshold_idle_movement=args.environment_config.get('thresholds', {}).get('idle_movement', 0),
        rewards=args.environment_config.get('rewards', None),
        
        # agent parameters
        use_dqn=args.agent_config.get('use_dqn', True),
        learning_rate=args.agent_config.get('learning_rate', 0.0005),
        gamma=args.agent_config.get('gamma', 0.9),
        batch_size=args.agent_config.get('batch_size', 32),
        epsilon_start=args.agent_config.get('epsilon_start', 1.0),
        min_epsilon=args.agent_config.get('min_epsilon', 0.05),
        epsilon_decay=args.agent_config.get('epsilon_decay', 0.99),
        memory_size=args.agent_config.get('memory_size', 10000),
        target_update_freq=args.agent_config.get('target_update_freq', 100),
        
        # world model parameters
        world_model_path=args.world_model_config.get('model_path', 'mini_world_model'),
        world_model_n_categories=args.world_model_config.get('n_categories', 32),
        world_model_hidden_size=args.world_model_config.get('hidden_size', 64),
        
        #training parameters
        episodes=args.training_config.get('episodes', 50),
        max_steps=args.training_config.get('max_steps', 30),
        display_every=args.training_config.get('display_every', 5),
        headless=args.training_config.get('headless', False),
        model_name=args.training_config.get('model_name', 'q_table_pc'),
        experiment_name=args.training_config.get('experiment_name', 'ESP32_Robot_Navigation_2D'),
        output_dir=args.training_config.get('output_dir', ''),
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="model_default", help="Nome do experimento/modelo")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--headless", action='store_true', help="Executar sem renderização")
    parser.add_argument("--config", type=str, default=None, help="Caminho para arquivo de configuração (opcional)")
    args = parser.parse_args()
    
    config= None
    model_name = args.name
    
    print(f"🧪 Iniciando Experimento: {model_name}")
    model_save_name = f"{model_name}.pth" # O path completo será tratado na função save
    


    if args.config:
        config = load_config(os.path.join('./configs/', args.config + '.yaml'))
        model_name = config.get("model_name", model_name)
        
        train_args = TrainingArguments(
            environment_config=config.get("environment", {}),
            agent_config=config.get("agent", {}),
            world_model_config=config.get("world_model", {}),
            training_config=config.get("training", {})
        )
        
        AGENTE_TREINADO, AMBIENTE, METRICAS = train_agent_from_config(train_args)
        
        
        print("\n📊 Resultado Final:")
        print(f"Média de Recompensa Final: {np.mean(METRICAS['reward_history'][-10:]):.1f}")
        print("Modelo salvo em: q_table_pc.pth")
        print("Gráfico salvo em: loss_chart.png")
        print('Historico de recompensas: ')
        print(METRICAS['reward_history'][:50])
        

    else:
        AGENTE_TREINADO, AMBIENTE, METRICAS = train_agent(
            model_name=model_name, # Passa o nome limpo
            episodes=args.episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            headless=args.headless,
            display_every=1
        )

        print("\n📊 Resultado Final:")
        print(f"Média de Recompensa Final: {np.mean(METRICAS['reward_history'][-10:]):.1f}")
        print("Modelo salvo em: q_table_pc.pth")
        print("Gráfico salvo em: loss_chart.png")
        print('Historico de recompensas: ')
        print(METRICAS['reward_history'][:50])