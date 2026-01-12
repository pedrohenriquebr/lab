import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mss
import argparse
import os
import time

# Imports do Projeto
from esp32robot.agents import VisualWorldModel 
from esp32robot.utils import load_config

# Configurações Fixas
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MONITOR = {"top": 100, "left": 100, "width": 600, "height": 600}

# ================= CLASSES DE FONTE =================
class ScreenSource:
    def __init__(self, monitor):
        self.sct = mss.mss()
        self.monitor = monitor

    def get_frame(self):
        img_raw = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)

    def close(self):
        self.sct.close()

class VideoSource:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir: {video_path}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret: # Loop infinito
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame 

    def close(self):
        self.cap.release()

# ================= VISUALIZAÇÃO =================
def visualize(args):
    print(f"👁️ Inicializando Visualizador no device: {DEVICE}")
    
    # 1. Configuração
    config_path = f"configs/{args.config}.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config não encontrada: {config_path}")
        return

    cfg = load_config(config_path)
    model_cfg = cfg['model']
    input_shape = tuple(model_cfg['input_shape']) # (3, H, W)
    img_h, img_w = input_shape[1], input_shape[2]
    
    print(f"⚙️ Modelo: {input_shape} | Actions: {model_cfg['n_actions']}")

    # 2. Fonte de Imagem
    source = VideoSource(args.video) if args.video else ScreenSource(MONITOR)

    # 3. Modelo
    model = VisualWorldModel(
        input_shape=input_shape,
        n_actions=model_cfg['n_actions'],
        n_categorias=model_cfg['n_categories'],
        hidden_size=model_cfg['hidden_size']
    ).to(DEVICE)
    
    # 4. Pesos
    if args.model:
        model_path = args.model
    else:
        # Pega o caminho do config
        model_path = os.path.join(cfg['training']['output_dir'], "last_visual_model.pth")

    if os.path.exists(model_path):
        print(f"✅ Carregando pesos: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            # strict=False permite carregar pesos mesmo se tiver pequenas diferenças (ex: inverse head nova)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # É um checkpoint de treino (Last)
                weights = checkpoint['model_state_dict']
            else:
                # É apenas o arquivo de pesos (Best)
                weights = checkpoint
            
            model.load_state_dict(weights, strict=False)
        except Exception as e:
            print(f"❌ Erro de pesos: {e}")
    else:
        print(f"⚠️ Pesos não encontrados! Rodando aleatório.")
    
    model.eval()

    # 5. Setup Gráfico
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_real, ax_dream, ax_latent = axes
    
    # Pre-configuração dos eixos para ser mais rápido
    for ax in [ax_real, ax_dream]:
        ax.axis('off')

    print("🎥 Visualizando... (Ctrl+C para sair)")
    
    try:
        while True:
            # A. Captura
            frame_bgr = source.get_frame()
            if frame_bgr is None: continue 
            
            # B. Preprocessamento (Resize + Tensor)
            frame_resized = cv2.resize(frame_bgr, (img_w, img_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            tensor = torch.from_numpy(frame_rgb).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE) # (1, C, H, W)

            # C. Inferência
            with torch.no_grad():
                # CORREÇÃO: hard=False e Temperature baixa para suavidade
                recon, z_dist, _, _ = model(tensor, hard=False, temperature=0.1)

            # D. Pós-processamento
            img_dream = recon[0].permute(1, 2, 0).cpu().numpy()
            probs = z_dist[0].cpu().numpy()
            cat_idx = np.argmax(probs)

            # E. Plotagem
            ax_real.imshow(frame_rgb)
            ax_real.set_title("Input Real")
            
            ax_dream.imshow(np.clip(img_dream, 0, 1))
            ax_dream.set_title(f"Reconstrução (Decoder)\nConceito Principal: #{cat_idx}")
            
            ax_latent.clear()
            ax_latent.bar(range(len(probs)), probs, color='purple')
            ax_latent.set_ylim(0, 1.0)
            ax_latent.set_title("Ativação Latente (Encoder)")

            plt.pause(0.001) # Mínimo possível para atualizar a tela

    except KeyboardInterrupt:
        print("\n🛑 Fim.")
    finally:
        source.close()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="carracing_pretrain")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    visualize(args)