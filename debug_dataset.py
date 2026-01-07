import cv2
import numpy as np
import torch
from esp32robot.data import MineRLDataset

# Configuração
ROOT_PATH = "datasets/minerl/train"
OUTPUT_VIDEO = "debug_action_check.mp4"
NUM_FRAMES_TO_CHECK = 1000  # Quantos frames analisar

# Mapeamento reverso para texto
ACTION_NAMES = {
    0: "STOP",
    1: "FORWARD",
    2: "BACK",
    3: "LEFT",
    4: "RIGHT"
}

def tensor_to_cv2(tensor):
    """Converte Tensor (C, H, W) -> Numpy BGR para OpenCV"""
    # Desfaz a normalização (assumindo que estava 0-1)
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    # RGB para BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def main():
    print("🕵️ Iniciando auditoria do Dataset...")
    
    # Instancia o dataset (usamos max_videos=1 para ser rápido)
    dataset = MineRLDataset(ROOT_PATH, img_size=128, max_videos=1) 
    
    # Prepara gravador de vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (128, 128))
    
    stats = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    print(f"🎥 Gerando vídeo de diagnóstico: {OUTPUT_VIDEO}")
    
    count = 0
    # O dataset retorna: (obs, action, next_obs)
    for obs, action_id, _ in dataset:
        if count >= NUM_FRAMES_TO_CHECK: break
        
        # Converte imagem
        frame = tensor_to_cv2(obs)
        
        # Pega nome da ação
        act_idx = int(action_id)
        act_name = ACTION_NAMES.get(act_idx, "UNKNOWN")
        stats[act_idx] += 1
        
        # Desenha na tela
        # Se for STOP, pinta de vermelho, se for FWD pinta de verde
        color = (0, 0, 255) if act_idx == 0 else (0, 255, 0)
        
        cv2.putText(frame, f"Act: {act_idx}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, act_name, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        out.write(frame)
        count += 1
        
        if count % 100 == 0:
            print(f"Processados {count} frames...")

    out.release()
    print("\n📊 ESTATÍSTICAS DE AÇÃO (Distribuição):")
    total = sum(stats.values())
    for k, v in stats.items():
        perc = (v / total) * 100 if total > 0 else 0
        print(f"  {ACTION_NAMES[k]}: {v} frames ({perc:.1f}%)")
        
    print(f"\n✅ Verifique o arquivo '{OUTPUT_VIDEO}' visualmente!")

if __name__ == "__main__":
    main()