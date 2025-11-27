from esp32robot.agents import QAgent2D
from esp32robot.config import MODEL_PATH, MODELS_DIR
from esp32robot.simulation import Esp322DEnv


def run_evaluation(model_name="q_table_pc", episodes=5, delay=0.05):
    """
    Carrega um modelo treinado e roda visualmente sem treinar.
    """
    import os
    import time
    import torch

    print(f"\n🎬 INICIANDO MODO DE AVALIAÇÃO (VISUAL)")
    
    # 1. Cria o ambiente com renderização HUMAN (Janela PyGame)
    env = Esp322DEnv(render_mode="human")
    
    # 2. Cria o agente (mesma configuração do treino)
    agent = QAgent2D(env.action_space, use_dqn=True)
    
    # 3. Carrega o Modelo
    model_file_path = f"{MODELS_DIR}/{model_name}.pth"
    if os.path.exists(model_file_path):
        agent.load(model_file_path)
    else:
        print(f"❌ Erro: Arquivo {model_name}.pth não encontrado!")
        env.close()
        return

    # 4. CONFIGURAÇÃO CRÍTICA PARA AVALIAÇÃO
    agent.epsilon = 0.0       # 0% de aleatoriedade (Pura inteligência)
    agent.policy_net.eval()   # Coloca o PyTorch em modo de inferência
    
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
        print("✅ Janela fechada.")


if __name__ == "__main__":
# ==========================================
    # EXECUTE ISTO PARA VER O ROBÔ ANDANDO
    # ==========================================
    run_evaluation(model_name='q_table_pc_1', episodes=10)