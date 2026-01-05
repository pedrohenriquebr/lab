from esp32robot.agents import QAgent2D
from esp32robot.config import MODELS_DIR
from esp32robot.simulation import Esp322DEnv
from esp32robot.mappers import map_config_params_to_dict

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os

from esp32robot.utils import ConfigurationParameters, load_config

def run_evaluation(model_name="q_table_pc", episodes=5, 
                    delay=0.05, 
                    latency_steps=0, 
                    stack_size=4, 
                    env_type='maze',
                    threshold_blocked_front:float=0,
                    threshold_corner:float=0,
                    threshold_proximity_danger:float=0,
                    threshold_idle_movement:float=0,
                    rotation_penalty=0.1,
                    rewards=None,
                    output_dir: str  = '',                    
                    debug: bool = False,
                    world_model_path: str = 'mini_world_model',
                    world_model_n_categories: int = 32,
                    world_model_hidden_size: int = 64,
                    headless=True, show_results=False) -> tuple[float, dict[int, int], float]: 
    """
    Carrega um modelo treinado e roda visualmente sem treinar.
    Agora inclui visualização do World Model (Sonho vs Realidade).
    """
    render_mode = "human" if not headless else None
    if not headless:
        print(f"\n🎬 INICIANDO MODO DE AVALIAÇÃO (VISUAL)")

    # 1. Cria o ambiente
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
    # 2. Cria o agente (O init dele já carrega o World Model se existir o arquivo mini_world_model.pth)
    # Certifique-se de que o arquivo .pth do world model está na raiz ou onde o agente espera
    agent = QAgent2D(env.action_space, env.observation_space, use_dqn=True,  
                            world_model_path=world_model_path,
                            world_model_n_categories=world_model_n_categories,
                            world_model_hidden_size=world_model_hidden_size,
                            debug=debug)
    
    if output_dir is None or output_dir == '':
        model_file_path = str(MODELS_DIR / f"{model_name}.pth")
    else:
        model_file_path = str(os.path.join(os.path.join(os.getcwd(), output_dir), f"{model_name}.pth"))

    print(f"🔄 Carregando agente de: {model_file_path}")

    if os.path.exists(model_file_path):
        agent.load(model_file_path)
    else:
        print(f"❌ Erro: Arquivo {model_name}.pth não encontrado!")
        env.close()
        return 0, {}, 0

    # 4. CONFIGURAÇÃO PARA AVALIAÇÃO
    agent.epsilon = 0.0       # 0% de aleatoriedade
    if hasattr(agent, 'policy_net'):
        agent.policy_net.eval()
    
    # --- CONFIGURAÇÃO DA VISUALIZAÇÃO DO WORLD MODEL ---
    if not headless:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # Gráfico 1: Reconstrução (Visão Real vs Sonho)
        # Cria linhas vazias iniciais
        x_axis = np.arange(env.observation_space.shape[0])
        line_real, = ax1.plot(x_axis, np.zeros_like(x_axis), label='Real (Input)', color='blue', linewidth=2)
        line_dream, = ax1.plot(x_axis, np.zeros_like(x_axis), label='Sonho (Reconstrução)', color='red', linestyle='--', linewidth=2)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_title("Visão do Robô: Real vs Imaginado (WM)")
        ax1.legend(loc='upper right')
        ax1.grid(True)

        # Gráfico 2: Estado Latente (Categorias)
        # Assume 32 categorias (pegando do agente)
        n_cats = agent.n_categorias
        bar_probs = ax2.bar(range(n_cats), np.zeros(n_cats), color='purple')
        ax2.set_ylim(0, 1.1)
        ax2.set_title("Ativação do Espaço Latente (Conceitos)")
        ax2.set_xlabel("ID da Categoria")
        
        plt.tight_layout()

    stats: dict[str, object] = {
        "rewards": [],
        "actions": {0:0, 1:0, 2:0, 3:0, 4:0},
        "success_counts": 0
    }
    
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            step = 0
            
            print(f"▶️ Episódio {ep+1}/{episodes} iniciou...")
            
            while not done:
                # --- VISUALIZAÇÃO WORLD MODEL ---
                if not headless:
                    # Passa a observação pelo World Model para ver o que ele "pensa"
                    #obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
                    
                    with torch.no_grad():
                        # O forward do seu DiscreteWorldModel retorna: 
                        # reconstrucao, z_dist, z_future_logits, logits
                        rec, z_dist, _, logits = agent.wm(obs_tensor)
                        
                        # Pegamos probabilidades para o gráfico de barras
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                        rec_np = rec.cpu().numpy()[0]
                        
                        # Categoria vencedora
                        cat_idx = np.argmax(probs)

                    # Atualiza Gráfico 1 (Linhas)
                    line_real.set_ydata(obs)
                    line_dream.set_ydata(rec_np)
                    
                    # Atualiza Gráfico 2 (Barras)
                    for rect, h in zip(bar_probs, probs):
                        rect.set_height(h)
                        # Pinta de vermelho a vencedora
                        rect.set_color('red' if h > 0.5 else 'purple')
                    
                    ax1.set_title(f"Ep {ep+1} | Step {step} | Latente Ativo: #{cat_idx}")
                    
                    # Renderiza
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                # --- AGENTE AGE ---
                action = agent.get_action(obs)
                stats["actions"][action] = stats["actions"].get(action, 0) + 1
                
                next_obs, reward, done, _, _ = env.step(action)
                
                obs = next_obs
                total_reward += reward
                step += 1
                
                # Delay visual
                if not headless:
                    # Se delay for muito pequeno, o matplotlib pode travar, então garantimos um mínimo
                    actual_delay = max(delay, 0.001)
                    time.sleep(actual_delay)
                
                if step > 500:
                    if not headless: print("   ⚠️ Timeout.")
                    break
           
            stats['success_counts'] += 1 if not done and reward > 0 else 0 # type: ignore
            stats['rewards'].append(total_reward) # type: ignore
            
            if not headless:
                print(f"   🏁 Fim! Reward: {total_reward:.1f}")

    except KeyboardInterrupt:
        print("\n⛔ Avaliação interrompida!")
    finally:
        env.close()
        if not headless:
            plt.ioff()
            plt.close('all')
        
        if headless and not show_results:
            return np.median(stats["rewards"]), stats["actions"], stats["success_counts"]/episodes # type: ignore
    
        # --- RELATÓRIO FINAL ---
        print("\n📊 RELATÓRIO FINAL")
        total_actions = sum(stats["actions"].values()) # type: ignore
        action_names = {0:'Stop', 1:'Fwd', 2:'Back', 3:'Left', 4:'Right'}
        
        for action, count in stats["actions"].items(): # type: ignore
            pct = (count / total_actions) * 100 if total_actions > 0 else 0
            print(f"   {action_names[action]}: {count} ({pct:.1f}%)")
            
        avg_reward = np.mean(stats["rewards"]) # type: ignore
        success_rate = (stats["success_counts"] / episodes) * 100 # type: ignore
        print(f"🏆 Taxa de Sucesso: {success_rate:.1f}%")
        print(f"📈 Recompensa Média: {avg_reward:.1f}")
        
        return np.median(stats["rewards"]), stats["actions"], success_rate/100 # type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="model_default", help="Nome do modelo (.pth)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, required=True, default=0.01, help="Delay entre frames (s)")
    parser.add_argument("--headless", action="store_true", help="Rodar sem gráficos")
    parser.add_argument("--config", type=str, default=None, help="Caminho para arquivo de configuração (opcional)")
    
    args = parser.parse_args()
    
    
    
    config= None
    model_name = args.name
    
    print(f"🧪 Iniciando Experimento: {model_name}")
    model_save_name = f"{model_name}.pth" # O path completo será tratado na função save
    


    if args.config:
        config = load_config(os.path.join('./configs/', args.config + '.yaml'))
        model_name = config.get("model_name", model_name)
        
        eval_args = ConfigurationParameters(
            environment_config=config.get("environment", {}),
            agent_config=config.get("agent", {}),
            world_model_config=config.get("world_model", {}),
            training_config=config.get("training", {})
        )
        
        params  = map_config_params_to_dict(eval_args)
        eval_params = [ 'model_name','episodes','latency_steps','stack_size',
                       'env_type','threshold_blocked_front','threshold_corner',
                       'threshold_proximity_danger','threshold_idle_movement',
                       'rotation_penalty','rewards','output_dir','debug',
                       'world_model_path','world_model_n_categories',
                       'world_model_hidden_size','headless']
        
        filtered_params = {k: v for k, v in params.items() if k in eval_params}
        filtered_params['delay'] = args.delay
        filtered_params['episodes'] = args.episodes
        filtered_params['headless'] = args.headless
        print(f"⚙️ Parâmetros de Avaliação carregados do arquivo de configuração:")
        for k, v in filtered_params.items():
            print(f"   - {k}: {v}")
        
        run_evaluation(**filtered_params)
        
    else:
        # Exemplo de uso: python evaluate.py --name meu_agente --episodes 3
        run_evaluation(
            model_name=args.name, 
            episodes=args.episodes, 
            delay=args.delay, 
            headless=args.headless
        )