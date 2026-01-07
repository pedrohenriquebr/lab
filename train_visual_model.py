import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# Imports do Projeto
from esp32robot.agents import VisualWorldModel
from esp32robot.data import create_dataloader, MineRLDataset
from esp32robot.utils import load_config

# Configuração de Backend para não abrir janela (headless) - Ideal para servidores/treino longo
plt.switch_backend('Agg')

# ==============================================================================
# 1. FUNÇÕES AUXILIARES DE VISUALIZAÇÃO E VALIDAÇÃO
# ==============================================================================

def save_snapshot(model, val_loader, device, epoch, output_dir, temp):
    """
    Gera uma imagem comparativa (Real vs Sonho) e salva no disco.
    Serve como um 'Relatório Visual' automático.
    """
    model.eval()
    
    # Pega um batch fixo (sempre o primeiro do val_loader para comparação justa)
    iterator = iter(val_loader)
    try:
        obs, _, _ = next(iterator)
    except StopIteration:
        return

    obs = obs.to(device)
    
    with torch.no_grad():
        # Roda o modelo. Usamos hard=False para ver a incerteza se houver.
        # Importante: Passar a temperatura atual para ver o comportamento real.
        recon, z_dist, _, _ = model(obs, hard=False, temperature=temp)
    
    # Pega a primeira imagem do batch para plotar
    # PyTorch (C, H, W) -> Matplotlib (H, W, C)
    img_real = obs[0].permute(1, 2, 0).cpu().numpy()
    img_dream = recon[0].permute(1, 2, 0).cpu().numpy()
    
    # Probabilidades do Latente
    probs = z_dist[0].cpu().numpy()
    cat_idx = np.argmax(probs)

    # Cria a figura
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(np.clip(img_real, 0, 1))
    axes[0].set_title("Realidade (Input)")
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(img_dream, 0, 1))
    axes[1].set_title(f"Sonho (Decoder)\nConceito: #{cat_idx}")
    axes[1].axis('off')
    
    axes[2].bar(range(len(probs)), probs, color='purple')
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title(f"Ativação Latente (Temp: {temp:.2f})")
    axes[2].set_xlabel("Categorias")
    
    # Salva
    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(save_path)
    plt.close()
    # print(f"📸 Snapshot salvo: {save_path}") # Comentado para não poluir o tqdm

def validate(model, val_loader, device, cfg, current_temp):
    """
    Calcula o Loss no conjunto de validação (dados que o modelo nunca viu).
    Isso diz se o modelo está decorando (overfitting) ou aprendendo.
    """
    model.eval()
    total_loss = 0
    steps = 0
    
    # Valida em um subset para ser rápido (ex: 50 batches)
    max_val_steps = 50 
    
    with torch.no_grad():
        for batch_obs, batch_action, batch_next_obs in val_loader:
            batch_obs = batch_obs.to(device)
            batch_action = batch_action.to(device)
            batch_next_obs = batch_next_obs.to(device)
            
            # Forward
            recon, z_dist, z_future_logits, _ = model(batch_obs, action=batch_action, hard=False, temperature=current_temp)
            
            # Recria targets
            target_probs = model.get_latent_probs(batch_next_obs)
            target_indices = torch.argmax(target_probs, dim=1)

            loss_rec = nn.MSELoss()(recon, batch_obs)
            loss_pred = nn.CrossEntropyLoss()(z_future_logits, target_indices)
            
            loss = (cfg['training']['beta_rec'] * loss_rec) + \
                   (cfg['training']['beta_pred'] * loss_pred)
            
            total_loss += loss.item()
            steps += 1
            
            if steps >= max_val_steps: break
            
    return total_loss / max(1, steps)

# ==============================================================================
# 2. LOOP DE TREINO
# ==============================================================================

def train_one_epoch(model, dataloader, optimizer, device, cfg, current_epoch, temp_scheduler):
    model.train()
    total_loss_rec = 0
    total_loss_pred = 0
    steps = 0
    
    # Calcula temperatura atual (Annealing)
    # Começa alta (1.0) para explorar, desce para baixa (0.1) para decidir
    start_temp, end_temp, decay_epochs = temp_scheduler
    progress = min(1.0, current_epoch / decay_epochs)
    current_temp = start_temp - (progress * (start_temp - end_temp))
    
    # Estima total de batches para a barra de progresso
    try:
        total_samples = dataloader.dataset.total_samples
        total_batches = total_samples // dataloader.batch_size
    except:
        total_batches = len(dataloader) # Fallback

    pbar = tqdm(dataloader, total=total_batches, desc=f"Ep {current_epoch} (T={current_temp:.2f})", unit="batch")
    
    total_loss_inv = 0 

    for batch_obs, batch_action, batch_next_obs in pbar:
        batch_obs = batch_obs.to(device)
        batch_action = batch_action.to(device)
        batch_next_obs = batch_next_obs.to(device)
        
        # 1. Forward Pass (Frame Atual -> Reconstrução + Predição Futura)
        recon, z_dist, z_future_logits, _ = model(batch_obs, action=batch_action, hard=False, temperature=current_temp)
        
        # 2. Forward Pass no Frame Futuro (Para Dinâmica Inversa)
        # Precisamos saber qual é o "Código Latente Real" do próximo frame
        # Chamamos o modelo no next_obs. Não precisamos de ação nem reconstrução aqui, só o z_dist.
        _, z_next_dist, _, _ = model(batch_next_obs, hard=False, temperature=current_temp)

        # 3. Dinâmica Inversa: Adivinhe a ação!
        # Passamos o Z atual e o Z futuro (REAL)
        pred_action_logits = model.predict_action_inverse(z_dist, z_next_dist)

        # 4. Cálculo das Perdas
        
        # Target do Futuro (para o Predictor normal)
        with torch.no_grad():
            target_probs = model.get_latent_probs(batch_next_obs)
            target_indices = torch.argmax(target_probs, dim=1)

        loss_rec = nn.MSELoss()(recon, batch_obs)
        loss_pred = nn.CrossEntropyLoss()(z_future_logits, target_indices)
        
        # --- NOVA LOSS INVERSA ---
        # Compara a ação que o modelo "chutou" com a ação que realmente aconteceu (batch_action)
        # batch_action deve ser indices (long). Se for float, convertemos.
        loss_inv = nn.CrossEntropyLoss()(pred_action_logits, batch_action.long())

        # Entropia (Opcional, mantém o que você já tinha)
        probs = torch.softmax(z_future_logits, dim=1)
        avg_probs = torch.mean(probs, dim=0) 
        entropy_loss = torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        beta_ent = cfg['training'].get('beta_entropy', 0.1)
        beta_inv = cfg['training'].get('beta_inverse', 1.0)
        
        # --- SOMA TOTAL ---
        # Adicionei um peso beta_inv (pode ser 1.0 ou 0.5)
        loss = (cfg['training']['beta_rec'] * loss_rec) + \
               (cfg['training']['beta_pred'] * loss_pred) + \
               (beta_inv * loss_inv) + \
               (beta_ent * entropy_loss)
        
        # 5. Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Logs
        total_loss_rec += loss_rec.item()
        total_loss_pred += loss_pred.item()
        total_loss_inv += loss_inv.item() # Log novo
        steps += 1
        
        # Atualiza barra com a nova métrica 'Inv'
        pbar.set_postfix({
            'Rec': f"{loss_rec.item():.3f}", 
            'Pred': f"{loss_pred.item():.3f}",
            'Inv': f"{loss_inv.item():.3f}" 
        })
        
    avg_rec = total_loss_rec / max(1, steps)
    avg_pred = total_loss_pred / max(1, steps)
    
    # Retorna também o log da inversa se quiser printar no main
    return avg_rec, avg_pred, current_temp

# ==============================================================================
# 3. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Treino Profissional Automatizado")
    parser.add_argument("--config", type=str, default="minerl_pretrain")
    args = parser.parse_args()
    
    cfg = load_config(f"configs/{args.config}.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuração de Temperatura (Pode mover pro YAML depois)
    TEMP_START = 2.0
    TEMP_END = 0.1
    TEMP_DECAY_EPOCHS = 30
    temp_scheduler = (TEMP_START, TEMP_END, TEMP_DECAY_EPOCHS)

    # --- PREPARAÇÃO DE DIRETÓRIOS ---
    # Cria pasta de logs e snapshots
    exp_name = cfg['training'].get('experiment_name', 'default_run')
    log_dir = os.path.join("logs", exp_name)
    snap_dir = os.path.join(log_dir, "snapshots")
    model_dir = cfg['training']['output_dir']
    
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"📁 Logs salvos em: {log_dir}")
    print(f"📸 Snapshots salvos em: {snap_dir}")

    # --- DATASETS (SPLIT AUTOMÁTICO) ---
    print("📊 Preparando Datasets...")
    # Carrega dataset "mestre" para depois dividir
    # Nota: Precisamos instanciar MineRLDataset diretamente para acessar a lista de vídeos
    max_videos = cfg['dataset'].get('max_videos', None)
    
    # 1. Cria o dataset completo apenas para ver os vídeos
    full_dataset_ref = MineRLDataset(cfg['dataset']['train_path'], max_videos=max_videos)
    all_videos = full_dataset_ref.videos
    
    # 2. Divide: 90% Treino, 10% Validação
    total_videos = len(all_videos)
    val_count = max(1, int(total_videos * 0.1))
    train_count = total_videos - val_count
    
    train_videos = all_videos[:train_count]
    val_videos = all_videos[train_count:]
    # write to log file which videos are in train and val
    with open(os.path.join(log_dir, "data_split.txt"), "w") as f:
        f.write(f"Total Videos: {total_videos}\n")
        f.write(f"Train Videos ({len(train_videos)}):\n")
        for vid in train_videos:
            f.write(f"  {vid}\n")
        f.write(f"\nValidation Videos ({len(val_videos)}):\n")
        for vid in val_videos:
            f.write(f"  {vid}\n")
    
    print(f"✅ Total Vídeos: {total_videos} | Treino: {len(train_videos)} | Validação: {len(val_videos)}")
    
    # 3. Cria Loaders Reais injetando a lista de vídeos cortada
    # Hack para usar a classe existente sem mudar muito: instanciamos e forçamos a lista de vídeos
    train_ds = MineRLDataset(cfg['dataset']['train_path'], img_size=cfg['dataset']['img_size'], videos=train_videos)
    # Recalcula total_samples para a barra de progresso (opcional, mas bom pra precisão)
    # (Para simplificar, vamos deixar o loader calcular ou estimar)

    val_ds = MineRLDataset(cfg['dataset']['train_path'], img_size=cfg['dataset']['img_size'], videos=val_videos)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg['training']['batch_size'])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg['training']['batch_size'])

    # --- MODELO ---
    model = VisualWorldModel(
        input_shape=tuple(cfg['model']['input_shape']),
        n_actions=cfg['model']['n_actions'],
        n_categorias=cfg['model']['n_categories'],
        hidden_size=cfg['model']['hidden_size']
    ).to(device)
    
    # Carrega Checkpoint
    save_path_best = os.path.join(model_dir, "best_visual_model.pth")
    save_path_last = os.path.join(model_dir, "last_visual_model.pth")
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    if os.path.exists(save_path_last):
        print(f"🔄 Carregando último checkpoint: {save_path_last}")
        checkpoint = torch.load(save_path_last, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"⏩ Retomando da Época {start_epoch}")

    # --- LOOP PRINCIPAL ---
    print(f"🚀 Iniciando treino automatizado...")
    
    try:
        for epoch in range(start_epoch, cfg['training']['epochs']):
            start_time = time.time()
            
            # 1. Treino
            avg_rec, avg_pred, curr_temp = train_one_epoch(
                model, train_loader, optimizer, device, cfg, epoch, temp_scheduler
            )
            
            # 2. Validação & Snapshot
            val_loss = validate(model, val_loader, device, cfg, curr_temp)
            save_snapshot(model, val_loader, device, epoch, snap_dir, curr_temp)
            
            elapsed = time.time() - start_time
            
            # 3. Log e Checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path_best)
            
            # Salva o "Last" com metadados para resume
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, save_path_last)
            
            status = "🏆 BEST" if is_best else ""
            print(f"✅ Ep {epoch} | Val Loss: {val_loss:.4f} | Train Rec: {avg_rec:.4f} | Train Pred: {avg_pred:.4f} | {status}")
            
    except KeyboardInterrupt:
        print("\n🛑 Treino pausado. Checkpoint salvo.")

if __name__ == "__main__":
    main()