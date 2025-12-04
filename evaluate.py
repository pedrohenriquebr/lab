from esp32robot.agents import QAgent2D
from esp32robot.config import MODEL_PATH, MODELS_DIR
from esp32robot.simulation import Esp322DEnv
import argparse

def run_evaluation(model_name="q_table_pc", episodes=5, delay=0.05):
    """
    Carrega um modelo treinado e roda visualmente sem treinar.
    """
    import os
    import time

    print(f"\n🎬 INICIANDO MODO DE AVALIAÇÃO (VISUAL)")
    
    # 1. Cria o ambiente com renderização HUMAN (Janela PyGame)
    env = Esp322DEnv(render_mode="human", env_type='circular_race_track')
    
    # 2. Cria o agente (mesma configuração do treino)
    agent = QAgent2D(env.action_space, use_dqn=True)
    
    # 3. Carrega o Modelo
    model_file_path = str(MODELS_DIR / f"{model_name}.pth")

    if os.path.exists(model_file_path):
        agent.load(model_file_path)
    else:
        print(f"❌ Erro: Arquivo {model_name}.pth não encontrado!")
        env.close()
        return

    # 4. CONFIGURAÇÃO CRÍTICA PARA AVALIAÇÃO
    agent.epsilon = 0.0       # 0% de aleatoriedade (Pura inteligência)
    agent.policy_net.eval()   # Coloca o PyTorch em modo de inferência
    actions = {}
    
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
                actions[action] = actions.get(action, 0) + 1
                
                # Executa no ambiente
                next_obs, reward, done, _, _ = env.step(action)
                
                # NÃO CHAMAMOS agent.update() AQUI!
                
                obs = next_obs
                total_reward += reward
                step += 1
                
                # Delay para o olho humano conseguir acompanhar
                time.sleep(delay)
                
                # Se demorar demais (loop infinito), corta
                if step > 500:
                    print("   ⚠️ Forçando fim do episódio (timeout).")
                    break
            
            print(f"   🏁 Fim! Reward Total: {total_reward:.1f} | Steps: {step}")
            time.sleep(1) # Pausa entre episódios

    except KeyboardInterrupt:
        print("\n⛔ Avaliação interrompida!")
    finally:
        env.close()
        print("\n🎬 Modo de Avaliação finalizado.")
        # formatar melhor a visualização com porcentagem
        total_actions = sum(actions.values())
        # FWD
        # BACK  
        # LEFT
        # RIGHT
        action_names = {
            0:'Stop', 1:'Fwd', 2:'Back', 3:'Left', 4:'Right'
        }
        print("📊 Estatísticas de Ações Executadas:")
        for action, count in actions.items():
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            print(f"   Ação {action_names[action]}: {count} vezes ({percentage:.2f}%)")
        print("✅ Janela fechada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="model_default", help="Nome do experimento/modelo")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.00, help="Delay entre passos em segundos")
    args = parser.parse_args()

    print(f"🧪 Iniciando Experimento: {args.name}")
    # ==========================================
    # EXECUTE ISTO PARA VER O ROBÔ ANDANDO
    # ==========================================
    run_evaluation(model_name=args.name, episodes=args.episodes, delay=args.delay)