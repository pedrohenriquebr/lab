import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm

# Imports do seu projeto
from esp32robot.agents import VisualWorldModel
from esp32robot.data import create_dataloader
from esp32robot.utils import load_config

# Config
plt.switch_backend('Agg')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_debug(img_tensor, recon_tensor, step, output_dir):
    """Salva uma imagem comparando Real vs Reconstruído"""
    # Pega o primeiro item do batch e converte pra numpy
    img_real = img_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    img_recon = recon_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    
    # Desnormaliza se necessário (assumindo 0-1)
    img_real = np.clip(img_real, 0, 1)
    img_recon = np.clip(img_recon, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_real)
    axes[0].set_title("Input Real")
    axes[0].axis('off')
    
    axes[1].imshow(img_recon)
    axes[1].set_title(f"Reconstrução (Step {step})")
    axes[1].axis('off')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"debug_{step:04d}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="carracing_pretrain")
    args = parser.parse_args()

    # 1. Carrega Config
    cfg = load_config(os.path.join(os.getcwd(), f".\\configs\\{args.config}.yaml"))
    print(f"🔧 Testando Arquitetura CNN com config: {args.config}")

    # 2. Inicializa Modelo (Ignora Predictor/Inverse, foca no Autoencoder)
    model = VisualWorldModel(
        input_shape=tuple(cfg['model']['input_shape']),
        n_actions=cfg['model']['n_actions'],
        n_categorias=cfg['model']['n_categories'],
        hidden_size=cfg['model']['hidden_size'],
        # Se você implementou o cnn_channels parametrizável, descomente abaixo:
        cnn_channels=cfg['model'].get('cnn_channels', 32)
    ).to(DEVICE)
    
    print(f"🧠 Modelo criado. Hidden: {cfg['model']['hidden_size']}, Cats: {cfg['model']['n_categories']}")

    # 3. Carrega UM Batch de dados
    # Vamos usar o loader normal, pegar o primeiro batch e fechar
    print("📂 Carregando um batch de dados para Overfitting...")
    dataloader = create_dataloader(cfg['dataset']['train_path'], batch_size=32, img_size=cfg['model']['input_shape'][1])
    
    # Pega o primeiro batch (formato sequencia: [B, Seq, C, H, W])
    data_iter = iter(dataloader)
    seq_frames, _ = next(data_iter)
    
    # Vamos pegar apenas o primeiro frame de cada sequencia para ser um batch de imagens simples
    # Shape: [32, 3, 96, 96]
    fixed_batch = seq_frames[:, 0, :, :, :].to(DEVICE)
    
    print(f"🖼️ Batch fixo carregado: {fixed_batch.shape}")
    print(f"   Média de cor do input: {fixed_batch.mean().item():.3f} (Se for muito baixo, o vídeo tá escuro!)")

    # 4. Loop de Treino (Overfitting Intencional)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("\n🚀 Iniciando Teste de Sanidade (Overfitting)...")
    print("O objetivo é zerar a Loss. Se não zerar, a arquitetura tem defeito.")
    
    pbar = tqdm(range(1000)) # 1000 passos no mesmo batch
    
    for step in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward (apenas autoencoder)
        # hard=False permite gradiente passar pelo Gumbel-Softmax
        recon, z_dist, _, _ = model(fixed_batch, hard=False, temperature=1.0)
        
        loss = criterion(recon, fixed_batch)
        
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'MSE': f"{loss.item():.5f}"})
        
        if step % 100 == 0:
            plot_debug(fixed_batch, recon, step, "logs/debug_cnn")

    print("\n✅ Teste finalizado. Verifique a pasta 'logs/debug_cnn'.")
    print("Se a última imagem for idêntica à original, sua CNN está perfeita.")

if __name__ == "__main__":
    main()