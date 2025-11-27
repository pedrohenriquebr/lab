import statistics
import cv2
import time

# ==========================================
# 4. LOOP PRINCIPAL DE OPERAÇÃO
# ==========================================
from esp32robot.config import COMMANDS, DELAY, MODEL_PATH, MODELS_DIR, ROBOT_IP, STREAM_URL
from esp32robot.controller import DeployedAgent, RealRobotController

history_executions: dict[str, list[float]] = {}
def run_real_robot(model_name='q_table_pc', epsilon=0.05, max_steps=100):
    """
    Executa o agente treinado no robô real
    
    Args:
        model_path: Caminho para o modelo treinado (sem .pth)
        epsilon: Taxa de exploração (0 = 100% greedy)
        max_steps: Número máximo de ações antes de parar
    """
    print("="*60)
    print("🤖 INICIANDO AGENTE NO ROBÔ REAL")
    print(f"IP: {ROBOT_IP}")
    print(f"Modelo: {model_name}.pth")
    print(f"Exploração: {epsilon*100}%")
    print("="*60)
    global history_executions
    history_executions.clear()  # <--- ADICIONE ISSO
    model_file_path = f"{MODELS_DIR/model_name}.pth"

    # Inicializa agente e controller
    agent = DeployedAgent(model_file_path, epsilon=epsilon, is_2d_model=True)
    controller = RealRobotController(COMMANDS, STREAM_URL)
    
    
    # Inicializa janela de debug
    # Inicializa janela com thread otimizada para Jupyter
    cv2.namedWindow('ESP32-CAM Debug', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ESP32-CAM Debug', 1400, 720) 

    cv2.startWindowThread()  # 🔴 OTIMIZA PERFORMANCE NO JUPYTER
    
    step_count = 0
    last_non_stop_command = time.time()
    
    try:
        while step_count < max_steps:
            print(f"\n📷 Step {step_count + 1}/{max_steps}")
            
            # 1. Captura frame
            frame = controller.get_frame()
            if frame is None:
                print("❌ Frame inválido, tentando novamente...")
                time.sleep(0.5)
                continue
            # pil_image = frame.rotate(180)
            
            # 2. Processa e toma decisão
            state, debug_img = agent.process_frame(frame)
            action = agent.get_action(state)
            
            if action !=0:
                last_non_stop_command  = time.time()
            
            cv2.imshow('ESP32-CAM Debug', debug_img)

            
            # 4. Executa ação no robô
            success = controller.send_command(action)
            if not success:
                print("⛔ Falha crítica. Parando!")
                break
            
            # 5. Log
            action_names = ["STOP", "FWD", "BWD", "LEFT", "RGHT"]
            print(f"🎬 Ação: {action_names[action]}")
            print(f"📊 State: {state} | Target Cells: {state.count(4)}")
            
            if action == 0:
                print(f'o modelo parou por conta própria após {(time.time() - last_non_stop_command)}s')
            
            time.sleep(1)
            controller.send_command(0)
            if action != 0:
                print(f'Foi necessário parar programaticamente após {time.time() - last_non_stop_command}s')
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⛔ Parada manual!")
                controller.send_command(0)  # Envia STOP
                break
            
            step_count += 1
            time.sleep(DELAY)
            
    except KeyboardInterrupt:
        print("\n⛔ Interrompido pelo usuário!")
    finally:
        # Para o robô e fecha
        print("\n🛑 Parando robô...")
        controller.send_command(0)
        cv2.destroyAllWindows()
        print("✅ Operação finalizada!")
        
                
        for key,value in history_executions.items():
            print(f"Estatísticas para função '{key}':")
            print('\tTempo médio: ', statistics.mean(value) * 1000, 'ms')
            print('\tMenor Tempo: ', min(value) * 1000, 'ms')
            print('\tMaior Tempo: ', max(value) * 1000, 'ms')
            print('\tMediana : ', statistics.median(value) * 1000, 'ms')
            print('='*80)
            print()

# ==========================================
# 5. EXECUTAR! (DESCOMENTE PARA USAR)
# ==========================================
# 🔴 MODO EXPLORAÇÃO: 5% aleatório, 95% política aprendida
# run_real_robot(model_path=MODEL_PATH, epsilon=0.05, max_steps=100)

# 🔴 MODO EXPLOTAÇÃO: 100% política aprendida (não explora)

if __name__ == "__main__":
    run_real_robot(model_name='q_table_pc_1', epsilon=0.0, max_steps=200)
