import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
import mss
import pygame
# Seus módulos
from esp32robot.agents import VisualWorldModel
from esp32robot.simulation import Esp322DEnv
from esp32robot.utils import load_config

# Configurações Fixas
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_visual_data(env, model, n_samples, img_size):
    print(f"📸 Coletando {n_samples} frames visuais...")
    
    latents = []
    images = [] # Vamos guardar a imagem original para plotar depois
    
    obs, _ = env.reset()
    
    # Vamos usar o renderizador interno do PyGame para pegar a imagem limpa
    # Se fosse vídeo externo, usaríamos captura de tela
    
    for i in range(n_samples):
        # 1. Captura a imagem do ambiente (Render manual)
        # O Env precisa estar em modo 'rgb_array' ou usamos o Pygame Surface direto
        if env.window is None:
            env.render() # Força criar janela
        
        # Pega pixels brutos do Pygame
        canvas = pygame.display.get_surface()
        img_str = pygame.image.tostring(canvas, "RGB")
        img_raw = np.frombuffer(img_str, dtype=np.uint8).reshape((600, 800, 3)) # Ajuste para seu window_size
        
        # Corta e Redimensiona (Simula o que a CNN vê)
        # Assumindo que o jogo é quadrado ou focado
        img_crop = img_raw[:, :600, :] # Pega quadrado esquerdo 600x600
        img_small = cv2.resize(img_crop, (img_size, img_size))
        
        # Prepara Tensor
        img_float = img_small.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # 2. Roda World Model
        with torch.no_grad():
            _, _, _, logits = model(img_tensor)
            vec = logits.cpu().numpy()[0]
            
        latents.append(vec)
        
        # Guarda imagem pequena para o gráfico (converte para 0..1 para matplotlib)
        images.append(img_small)
        
        # Ação aleatória
        action = env.action_space.sample()
        _, _, done, _, _ = env.step(action)
        if done: env.reset()
            
        if i % 50 == 0: print(f"   Coletado: {i}/{n_samples}", end='\r')

    return np.array(latents), images

def plot_images_scatter(x, y, images, title):
    print("🎨 Gerando 'Nuvem de Imagens' (Isso pode demorar um pouco)...")
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(x, y, alpha=0.1) # Pontos transparentes de fundo
    
    # Para não poluir, vamos plotar apenas uma amostra das imagens (ex: a cada 5 ou 10)
    step_plot = max(1, len(images) // 150) # Tenta mostrar ~150 imagens no total
    
    for i in range(0, len(images), step_plot):
        img = images[i]
        im = OffsetImage(img, zoom=0.5) # Zoom ajusta o tamanho da miniatura
        ab = AnnotationBbox(im, (x[i], y[i]), xycoords='data', frameon=False)
        ax.add_artist(ab)
        
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.show()

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Nome da config (ex: best_kamikaze_v1)")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    
    # 1. Carrega Config
    config_path = os.path.join('./configs/', args.config + '.yaml')
    cfg = load_config(config_path)
    
    # 2. Configurações extraídas
    img_size = 96 # Padrão ou extrair se tiver no yaml
    n_cats = cfg['world_model'].get('n_categories', 32)
    model_path = cfg['world_model'].get('model_path', 'visual_world_model.pth')
    
    # 3. Inicializa Env e Modelo
    env = Esp322DEnv(render_mode='human', env_type=cfg['environment']['env_type'])
    model = VisualWorldModel(input_shape=(3, img_size, img_size), n_categorias=n_cats).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("✅ Visual World Model carregado!")
    except:
        print("⚠️ Pesos não encontrados, usando aleatórios.")
        
    # 4. Coleta
    latents, imgs = collect_visual_data(env, model, args.samples, img_size)
    env.close()
    
    # 5. Redução t-SNE
    print("\n📐 Calculando t-SNE (2D)...")
    tsne = TSNE(n_components=2, perplexity=30, init='random')
    X_2d = tsne.fit_transform(latents)
    
    # 6. Plota a Mágica
    plot_images_scatter(X_2d[:, 0], X_2d[:, 1], imgs, 
                        f"Mapa Mental Visual (t-SNE)\nCada imagem está posicionada onde o robô 'pensa' que ela pertence")

if __name__ == "__main__":
    main()