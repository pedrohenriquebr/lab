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

# Configurações Fixas de Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Área de captura padrão (só usada se não passar vídeo)
# Pode ser movida para o YAML futuramente se quiser
MONITOR = {"top": 100, "left": 100, "width": 600, "height": 600}

# ================= CLASSES DE FONTE (Mesmas do Treino) =================
class ScreenSource:
    """Captura frames da tela em tempo real"""
    def __init__(self, monitor):
        self.sct = mss.mss()
        self.monitor = monitor

    def get_frame(self):
        img_raw = np.array(self.sct.grab(self.monitor))
        img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)
        return img_bgr

    def close(self):
        self.sct.close()

class VideoSource:
    """Lê frames de um arquivo de vídeo (com loop automático)"""
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")
        print(f"📂 Lendo vídeo: {video_path}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # Fim do vídeo? Rebobina!
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame 

    def close(self):
        self.cap.release()

# ================= VISUALIZAÇÃO =================
def visualize(args):
    print(f"👁️ Inicializando Visualizador no device: {DEVICE}")
    
    # 1. Carrega Configurações
    config_path = f"configs/{args.config}.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Erro: Arquivo de config não encontrado: {config_path}")
        return

    cfg = load_config(config_path)
    
    # Extrai parâmetros do modelo
    model_cfg = cfg.get('model', {})
    input_shape = tuple(model_cfg.get('input_shape', [3, 64, 64]))
    n_actions = model_cfg.get('n_actions', 5)
    n_categories = model_cfg.get('n_categories', 32)
    hidden_size = model_cfg.get('hidden_size', 128)
    
    # Define tamanho da imagem baseado na entrada do modelo (H, W)
    # input_shape é (Canais, Altura, Largura)
    img_h, img_w = input_shape[1], input_shape[2] 
    
    print(f"⚙️ Configuração carregada: {input_shape} input | {n_categories} categorias")

    # 2. Configura a Fonte
    if args.video:
        source = VideoSource(args.video)
    else:
        print(f"📷 Modo Tela: Capturando {MONITOR}")
        source = ScreenSource(MONITOR)

    # 3. Inicializa Modelo
    model = VisualWorldModel(
        input_shape=input_shape,
        n_actions=n_actions,
        n_categorias=n_categories,
        hidden_size=hidden_size
    ).to(DEVICE)
    
    # 4. Carrega Pesos
    # Se o usuário passou --model, usa ele. Se não, tenta montar o path do config.
    if args.model:
        model_path = args.model
    else:
        # Tenta inferir o caminho padrão definido no treino
        train_cfg = cfg.get('training', {})
        model_path = os.path.join(train_cfg.get('output_dir', 'models'), 
                                  f"{train_cfg.get('model_name', 'model')}.pth")

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"✅ Pesos carregados de: {model_path}")
        except Exception as e:
            print(f"❌ Erro ao carregar pesos: {e}")
            print("⚠️ Rodando com pesos aleatórios (apenas para debug visual)")
    else:
        print(f"⚠️ Arquivo não encontrado: {model_path}")
        print("⚠️ Rodando com pesos aleatórios (verifique se o treino terminou e salvou)")
    
    model.eval()

    # 5. Setup do Matplotlib
    plt.ion() # Modo interativo
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax_real, ax_dream, ax_latent = axes

    print("🎥 Rodando... (Ctrl+C no terminal para parar)")
    
    try:
        while True:
            # A. Pega Frame Bruto (BGR)
            frame_bgr = source.get_frame()
            if frame_bgr is None: continue 
            
            # B. Prepara para o Modelo
            # Resize dinâmico usando o tamanho definido no config
            frame_resized = cv2.resize(frame_bgr, (img_w, img_h))
            
            # BGR -> RGB e Normaliza
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_float = frame_rgb.astype(np.float32) / 255.0
            
            # (H, W, C) -> (C, H, W) -> (1, C, H, W)
            frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # C. Roda o Modelo (Sonha)
            with torch.no_grad():
                # O forward retorna: rec, z_dist, future, logits, pred_reward
                outputs = model(frame_tensor,hard=True)
                rec = outputs[0]     # Reconstrução
                z_dist = outputs[1]  # Distribuição Latente
                # future = outputs[2] # Previsão futura (se tivesse ação)
                # reward = outputs[4] # Recompensa prevista

            # D. Processa Saídas para Plotar
            # Tensor (1, C, H, W) -> Numpy (H, W, C)
            img_dream = rec.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Latente (1, 32) -> Numpy (32,)
            latent_probs = z_dist[0].cpu().numpy()
            cat_idx = np.argmax(latent_probs)

            # E. Atualiza Gráficos
            # 1. Real
            ax_real.clear()
            ax_real.imshow(frame_rgb)
            ax_real.set_title("Realidade (Input)")
            ax_real.axis('off')

            # 2. Sonho
            ax_dream.clear()
            ax_dream.imshow(img_dream)
            ax_dream.set_title(f"Sonho (Decoder)\nConceito Ativo: #{cat_idx}")
            ax_dream.axis('off')
            
            # 3. Código de Barras (Latente)
            ax_latent.clear()
            ax_latent.bar(range(len(latent_probs)), latent_probs, color='purple')
            ax_latent.set_ylim(0, 1.1)
            ax_latent.set_title("O que o robô 'entendeu' (Encoder)")
            ax_latent.set_xlabel("ID da Categoria")

            # Atualiza janela
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Encerrando visualização...")
    finally:
        source.close()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizador de Sonhos do World Model")
    
    # Argumento principal: qual config usar?
    parser.add_argument("--config", type=str, default="minerl_pretrain", 
                        help="Nome do arquivo de configuração (sem .yaml) em configs/")
    
    # Opcionais para sobrescrever comportamento
    parser.add_argument("--video", type=str, default=None, 
                        help="Caminho do vídeo para testar. Se vazio, captura a tela.")
    
    parser.add_argument("--model", type=str, default=None, 
                        help="Caminho manual do .pth (se quiser forçar um arquivo específico)")
    
    args = parser.parse_args()
    visualize(args)