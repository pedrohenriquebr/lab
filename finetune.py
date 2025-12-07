import argparse
import torch
import mlflow
import os
import time
import numpy as np
from contextlib import nullcontext
from datetime import datetime, timedelta

from esp32robot.simulation import Esp322DEnv
from esp32robot.agents import QAgent2D
from esp32robot.config import MODELS_DIR, COLAB_MODE
from IPython.display import clear_output

def run_finetuning(base_model_name, new_model_name, episodes=100, max_steps=300, batch_size=64):
    
    # --- CONFIGURAÇÃO DE FINE-TUNING ---
    LR_FINETUNE = 0.00010880501821304784
    EPSILON_START = 0.2    # Pouca exploração (mas suficiente para sair de mínimos locais)
    EPSILON_DECAY = 0.95   # Cai rápido
    ENV_TYPE = 'default'      # Mapa difícil para forçar generalização
    
    # Penaliza levemente a DIREITA para corrigir vícios
    CORRECTION_WEIGHTS = {
        3:0,  # Left (Bônus leve)
        4:0   # Right (Multa leve)
    }

    # Configura MLflow
    mlflow.set_experiment("ESP32_Fine_Tuning")
    
    active_run = mlflow.active_run()
    if active_run:
        print(f"🔄 Anexando à Run existente: {active_run.info.run_id}")
        run_context = nullcontext()
    else:
        run_name = f"FT_{new_model_name}_{int(time.time())}"
        print(f"✨ Criando nova Run: {run_name}")
        run_context = mlflow.start_run(run_name=run_name)

    agent = None
    env = None
    
    with run_context:
        try:
            # 1. Inicializa Ambiente Difícil
            env = Esp322DEnv(render_mode=None, env_type=ENV_TYPE, rotation_penalty=0, 
                             action_weights=CORRECTION_WEIGHTS,
                             )

            # 2. Carrega o Agente
            agent = QAgent2D(env.action_space, use_dqn=True, batch_size=batch_size)
            
            # Carrega os pesos do modelo vencedor (BASE)
            base_path = str(MODELS_DIR / f"{base_model_name}.pth")
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Modelo base não encontrado: {base_path}")
                
            agent.load(base_path)
            # Copia para a rede alvo também
            if hasattr(agent, 'target_net'):
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            # 3. Ajusta Hiperparâmetros para "Cirurgia"
            agent.lr = LR_FINETUNE
            for g in agent.optimizer.param_groups: 
                g['lr'] = LR_FINETUNE
            
            agent.epsilon = EPSILON_START
            agent.epsilon_decay = EPSILON_DECAY
            
            mlflow.log_params({
                "base_model": base_model_name,
                "new_model": new_model_name,
                "lr_finetune": LR_FINETUNE,
                "env_type": ENV_TYPE,
                "correction": "Increase success rate"
            })
            
            print(f"🔧 Iniciando Fine-Tuning de '{base_model_name}' no mapa '{ENV_TYPE}'...")

            # 4. Loop de Treino
            for ep in range(episodes):
                obs, _ = env.reset()
                total_reward = 0
                episode_losses = []
                
                # Métricas extras
                action_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
                
                for step in range(max_steps):
                    action = agent.get_action(obs)
                    action_counts[action] += 1
                    
                    next_obs, reward, done, _, info = env.step(action)
                    
                    # Aplica Correção Manual de Viés (Se não implementou no env, faz aqui)
                    # Ex: Se virar pra direita, pune extra
                    if action == 4: reward -= 0.1
                    
                    loss = agent.update(obs, action, reward, next_obs)
                    if loss and loss != 0:
                        episode_losses.append(loss)
                    
                    obs = next_obs
                    total_reward += reward
                    if done: break
                
                # Atualizações de Fim de Episódio
                if agent.epsilon > agent.min_epsilon:
                    agent.epsilon *= agent.epsilon_decay
                
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                
                # Logs
                mlflow.log_metrics({
                    "ft_reward": total_reward,
                    "ft_loss": avg_loss,
                    "epsilon": agent.epsilon
                }, step=ep)
                
                # Log de Ações (Pra ver se parou de ir só pra direita)
                total_actions = sum(action_counts.values())
                if total_actions > 0:
                    mlflow.log_metrics({
                        "act_left_pct": action_counts[3] / total_actions,
                        "act_right_pct": action_counts[4] / total_actions
                    }, step=ep)

                if ep % 5 == 0:
                    if COLAB_MODE: clear_output(wait=True)
                    print(f"Ep {ep}/{episodes} | R: {total_reward:.1f} | L: {avg_loss:.4f} | Eps: {agent.epsilon:.2f}")

            # 5. Salva o Modelo Final
            save_path = str(MODELS_DIR / f"{new_model_name}.pth")
            agent.save(save_path)
            mlflow.log_artifact(save_path)
            
            print(f"✅ Modelo refinado salvo: {new_model_name}.pth")
            
        except KeyboardInterrupt:
            print("\n⛔ Fine-Tuning interrompido!")
            if agent:
                save_path = str(MODELS_DIR / f"{new_model_name}_interrupted.pth")
                agent.save(save_path)
                print("💾 Modelo parcial salvo.")
        
        finally:
            if env: env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Nome do modelo vencedor (sem .pth)")
    parser.add_argument("--new", type=str, default="production_v1", help="Nome do novo modelo")
    parser.add_argument("--episodes", type=int, default=100, help="Número de episódios para fine-tuning")
    args = parser.parse_args()

    run_finetuning(args.base, args.new, args.episodes)