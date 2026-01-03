from esp32robot.agents import QAgent2D
from esp32robot.config import MODEL_PATH, MODELS_DIR
from esp32robot.simulation import Esp322DEnv
import argparse
import numpy as np 

def run_evaluation(model_name="q_table_pc", episodes=5, delay=0.05, latency_steps=5, stack_size=8, headless=True, show_results=False) -> tuple[float, dict[int, int], float]: 
    """
    Carrega um modelo treinado e roda visualmente sem treinar.
    """
    import os
    import time
    mode = "human" if not headless else None
    if not headless:
        print(f"\n🎬 INICIANDO MODO DE AVALIAÇÃO (VISUAL)")

    # 1. Cria o ambiente com renderização HUMAN (Janela PyGame)
    env = Esp322DEnv(render_mode=mode, env_type='default', latency_steps=latency_steps, stack_size=stack_size)
    
    # 2. Cria o agente (mesma configuração do treino)
    agent = QAgent2D(env.action_space, env.observation_space, use_dqn=True)
    
    # 3. Carrega o Modelo
    model_file_path = str(MODELS_DIR / f"{model_name}.pth")
    
    print(f"🔄 Carregando modelo de: {model_file_path}")

    if os.path.exists(model_file_path):
        agent.load(model_file_path)
    else:
        print(f"❌ Erro: Arquivo {model_name}.pth não encontrado!")
        env.close()
        return 0, {}, 0

    # 4. CONFIGURAÇÃO CRÍTICA PARA AVALIAÇÃO
    agent.epsilon = 0.0       # 0% de aleatoriedade (Pura inteligência)
    agent.policy_net.eval()   # Coloca o PyTorch em modo de inferência
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
                # Pega a melhor ação possível (sem sorteio)
                action = agent.get_action(obs)
                stats["actions"][action] = stats["actions"].get(action, 0) + 1
                
                # Executa no ambiente
                next_obs, reward, done, _, _ = env.step(action)
                
                obs = next_obs
                total_reward += reward
                step += 1
                
                # Delay para o olho humano conseguir acompanhar
                if not headless:
                    time.sleep(delay)
                
                # Se demorar demais (loop infinito), corta
                if step > 500:
                    if not headless:
                        print("   ⚠️ Forçando fim do episódio (timeout).")
                    break
           
            stats['success_counts'] += 1 if not done and reward > 0 else 0 # type: ignore
            stats['rewards'].append(total_reward) # type: ignore
            
            if not headless:
                print(f"   🏁 Fim! Reward Total: {total_reward:.1f} | Steps: {step}")
                time.sleep(0.05) 

    except KeyboardInterrupt:
        print("\n⛔ Avaliação interrompida!")
    finally:
        env.close()
        
        if headless and not show_results:
            return np.median(stats["rewards"]), stats["actions"], stats["success_counts"]/episodes # type: ignore
    
        print("\n🎬 Modo de Avaliação finalizado.")
        # formatar melhor a visualização com porcentagem
        total_actions = sum(stats["actions"].values()) # type: ignore
        action_names = {
            0:'Stop', 1:'Fwd', 2:'Back', 3:'Left', 4:'Right'
        }
        print("📊 Estatísticas de Ações Executadas:")
        for action, count in stats["actions"].items(): # type: ignore
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            print(f"   Ação {action_names[action]}: {count} vezes ({percentage:.2f}%)")
            
        avg_reward = np.mean(stats["rewards"]) # type: ignore
        median_reward = np.median(stats["rewards"]) # type: ignore
        success_rate = (stats["success_counts"] / episodes) * 100 # type: ignore
        print(f"\n📈 Recompensa Média por Episódio: {avg_reward:.2f}")
        print(f"📈 Recompensa Mediana por Episódio: {median_reward:.2f}")
        print(f"🏆 Taxa de Sucesso: {success_rate:.2f}% ({stats['success_counts']} de {episodes})")
        
        print("✅ Janela fechada.")
        return median_reward, stats["actions"], success_rate/100 # type: ignore
            
            
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="model_default", help="Nome do experimento/modelo")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.00, help="Delay entre passos em segundos")
    parser.add_argument("--headless", action="store_true", help="Headless mode (no GUI)")
    parser.add_argument("--show-results", action="store_true", help="Show detailed results after evaluation")
    
    args = parser.parse_args()

    print(f"🧪 Iniciando Experimento: {args.name}")
    # ==========================================
    # EXECUTE ISTO PARA VER O ROBÔ ANDANDO
    # ==========================================
    run_evaluation(model_name=args.name, episodes=args.episodes, delay=args.delay, headless=args.headless, show_results=args.show_results)