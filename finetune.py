import argparse
import torch
import mlflow
import os
import time
import numpy as np
from contextlib import nullcontext

from esp32robot.simulation import Esp322DEnv
from esp32robot.agents import QAgent2D
from esp32robot.config import MODELS_DIR, COLAB_MODE
from IPython.display import clear_output

def run_finetuning(base_model_name, new_model_name, episodes=150, 
                   stack_size=4, latency_steps=5,
                   epsilon_start=0.3, epsilon_decay=0.98,
                   learning_rate=2e-5
                   ):
    
    # --- CONFIGURAÇÃO ---
    # Começa explorando para se adaptar ao lag no novo mapa
    epsilon_start = 0.3    
    epsilon_decay = 0.98   
    
    # Mapa difícil para provar que ele aprendeu a antecipar
    ENV_TYPE = 'default'      
    
    # Sem pesos manuais, deixa o Stacking resolver
    CORRECTION_WEIGHTS = {3: 0, 4: 0}

    
    # ... (Lógica de contexto MLflow igual) ...
    active_run = mlflow.active_run()
    if active_run:
        print(f"🔄 Anexando à Run existente: {active_run.info.run_id}")
        run_context = nullcontext()
    else:
        mlflow.set_experiment("ESP32_Fine_Tuning")
        run_name = f"FT_{new_model_name}_{int(time.time())}"
        run_context = mlflow.start_run(run_name=run_name)

    agent = None
    env = None
    
    with run_context:
        try:
            # 1. Inicializa com a MESMA arquitetura do treino
            env = Esp322DEnv(
                render_mode=None, 
                env_type=ENV_TYPE, 
                rotation_penalty=0.01,
                action_weights=CORRECTION_WEIGHTS,
                latency_steps=latency_steps, # Importante manter o lag
                stack_size=stack_size        # Importante manter a memória
            )

            # 2. Carrega Agente (O input_dim é calculado internamente pelo env/agent se configurado, 
            # ou precisamos passar. Vamos assumir que seu QAgent2D lida com isso ou você ajustou).
            # Se QAgent2D calcula input_dim = 3 * stack_size, passe o stack_size pra ele se necessário
            # ou garanta que ele pegue do env.observation_space.shape[0]
            
            # (Aqui assumo que QAgent2D pega o tamanho do env.observation_space automaticamente)
            agent = QAgent2D(env.action_space, env.observation_space, use_dqn=True, batch_size=32)
            
            # Carrega Pesos
            base_path = str(MODELS_DIR / f"{base_model_name}.pth")
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Modelo base não encontrado: {base_path}")
            
            # Tenta carregar. Se o stack_size estiver errado, vai dar erro aqui.
            try:
                agent.load(base_path)
            except RuntimeError as e:
                print(f"❌ ERRO DE SHAPE: Você tentou carregar um modelo com arquitetura diferente!")
                print(f"Confira se o 'stack_size' ({stack_size}) é o mesmo do treino original.")
                raise e

            if hasattr(agent, 'target_net'):
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            # Configura fine-tuning
            agent.lr = learning_rate
            for g in agent.optimizer.param_groups: g['lr'] = learning_rate
            agent.epsilon = epsilon_start
            agent.epsilon_decay = epsilon_decay
            
            mlflow.log_params({
                "base_model": base_model_name,
                "stack_size": stack_size,
                "latency_steps": latency_steps,
                "env_type": ENV_TYPE
            })
            
            print(f"🔧 Fine-Tuning: '{base_model_name}' (Stack={stack_size}, Lag={latency_steps})")

            # 3. Loop de Treino (Padrão)
            for ep in range(episodes):
                obs, _ = env.reset()
                total_reward = 0
                episode_losses = []
                
                # Aumente max_steps no Maze, pois com lag ele demora mais pra manobrar
                for step in range(400): 
                    action = agent.get_action(obs)
                    next_obs, reward, done, _, info = env.step(action)
                    
                    loss = agent.update(obs, action, reward, next_obs)
                    if loss and loss != 0: episode_losses.append(loss)
                    
                    obs = next_obs
                    total_reward += reward
                    if done: break
                
                if agent.epsilon > agent.min_epsilon:
                    agent.epsilon *= agent.epsilon_decay
                
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                
                mlflow.log_metrics({
                    "ft_reward": total_reward,
                    "ft_loss": avg_loss,
                    "epsilon": agent.epsilon
                }, step=ep)

                if ep % 5 == 0:
                    if COLAB_MODE: clear_output(wait=True)
                    print(f"Ep {ep}/{episodes} | R: {total_reward:.1f} | Eps: {agent.epsilon:.2f}")

            # 4. Salva (use pasta finetuned)
            FT_DIR = MODELS_DIR / "finetuned"
            if not FT_DIR.exists(): FT_DIR.mkdir()
            
            save_path = str(FT_DIR / f"{new_model_name}.pth")
            agent.save(save_path)
            mlflow.log_artifact(save_path)
            
            print(f"✅ Modelo refinado salvo em: {save_path}")
            
        except KeyboardInterrupt:
            print("\n⛔ Fine-Tuning interrompido!")
        finally:
            if env: env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Modelo Vencedor do Optuna")
    parser.add_argument("--new", type=str, default="production_v2_latency", help="Nome do novo modelo")
    parser.add_argument("--episodes", type=int, default=150)
    
    # NOVOS PARAMETROS QUE VOCÊ PRECISA PEGAR DO RESULTADO DO OPTUNA
    parser.add_argument("--stack", type=int, default=4, help="Stack Size usado no treino")
    parser.add_argument("--latency", type=int, default=5, help="Latency Steps usado no treino")
    
    args = parser.parse_args()

    run_finetuning(args.base, args.new, args.episodes, 
                   stack_size=args.stack, 
                   latency_steps=args.latency)