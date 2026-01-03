import argparse
import time
import cv2
from esp32robot.config import COMMANDS, STREAM_URL, MODELS_DIR, ROBOT_IP
from esp32robot.controller import DeployedAgent, RealRobotController, history_executions
import numpy as np
import pandas as pd

def run_real_robot(model_name, epsilon=0.0, max_steps=200):
    print("="*60)
    print("🤖 INICIANDO AGENTE NO ROBÔ REAL")
    print(f"IP: {ROBOT_IP}")
    print(f"Modelo: {model_name}")
    print("="*60)
    
    # Path do Modelo
    # Se for um modelo base, está em models/
    # Se for fine-tuned, pode estar em models/finetuned/
    # Vamos tentar achar
    path_base = MODELS_DIR / f"{model_name}.pth"
    path_ft = MODELS_DIR / "finetuned" / f"{model_name}.pth"
    
    final_path = str(path_ft) if path_ft.exists() else str(path_base)
    
    agent = DeployedAgent(final_path, epsilon=epsilon, is_2d_model=True)
    controller = RealRobotController(COMMANDS, STREAM_URL)
    data_log = []
    
    # Janela
    cv2.namedWindow('ESP32-CAM Debug', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ESP32-CAM Debug', 1200, 600)
    
    step = 0
    try:
        while step < max_steps:
            frame = controller.get_frame()
            if frame is None: 
                print("⚠️ Frame não recebido, tentando novamente...")
                continue
            
            # IA
            state, debug_img, current_sensors = agent.process_frame(frame)
            action, q_values = agent.get_action(state, return_q_values=True)
            data_log.append({
                "step": step,
                "sensors_state": current_sensors,
                "stacked_state": state,
                "q_values": q_values,
                "action": action,
                "timestamp": time.time()
            })
            
            # Display
            cv2.imshow('ESP32-CAM Debug', debug_img)
            
            # Comando
            act_name = ["STOP", "FWD", "BACK", "LEFT", "RIGHT"][action]
            print(f"Step {step} | Action: {act_name}")
            
            controller.send_command(action)
            time.sleep(0.1) # Pequeno delay
            controller.send_command(0) # Stop (Modo pulsado para controle fino)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            step += 1
            
    except KeyboardInterrupt:
        print("\nParando...")

    finally:
        controller.send_command(0)
        cv2.destroyAllWindows()
        for func_name, times in history_executions.items():
            median_time = np.median(times)
            print(f"[TIME STATS] {func_name}: {median_time * 1000:.1f} ms (median over {len(times)} calls)")
            
        import json
        df = pd.DataFrame(data_log)
        
        df.to_pickle("robot_data_log.pkl")
        print("📊 Dados salvos em robot_data_log.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Nome do arquivo do modelo (sem .pth)")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    
    run_real_robot(args.name, max_steps=args.steps)