import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# Imports do Projeto
from esp32robot.agents import VisualWorldModel
from esp32robot.data import create_dataloader, MineRLDataset
from esp32robot.utils import DynamicLossTuner, load_config

# Configuração de Backend para não abrir janela
plt.switch_backend('Agg')

# ==============================================================================
# 1. FUNÇÕES AUXILIARES DE VISUALIZAÇÃO E VALIDAÇÃO
# ==============================================================================

def save_snapshot(model, val_loader, device, epoch, output_dir, temp):
    model.eval()
    iterator = iter(val_loader)
    try:
        batch_frames, _ = next(iterator)
    except StopIteration:
        return

    obs = batch_frames[:, 0].to(device)
    
    with torch.no_grad():
        recon, z_dist, _, _ = model(obs, hard=False, temperature=temp)
    
    img_real = obs[0].permute(1, 2, 0).cpu().numpy()
    img_dream = recon[0].permute(1, 2, 0).cpu().numpy()
    probs = z_dist[0].cpu().numpy()
    cat_idx = np.argmax(probs)

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
    
    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(save_path)
    plt.close()

def validate(model, val_loader, device, cfg, current_temp):
    model.eval()
    total_loss = 0
    steps = 0
    max_val_steps = 50 
    
    with torch.no_grad():
        for batch_frames, batch_actions in val_loader:
            batch_frames = batch_frames.to(device)
            batch_actions = batch_actions.to(device)
            
            # 1. Reconstrução (Frame 0)
            first_frame = batch_frames[:, 0]
            recon, z_dist, _, _ = model(first_frame, hard=False, temperature=current_temp)
            loss_rec = nn.MSELoss()(recon, first_frame)
            
            # 2. Predição Multistep
            loss_pred_acc = 0
            current_z_dist = z_dist
            horizon = batch_actions.shape[1]
            
            for t in range(horizon):
                action = batch_actions[:, t]
                next_frame_real = batch_frames[:, t+1]
                z_next_logits = model.predict_next_from_dist(current_z_dist, action)
                target_probs = model.get_latent_probs(next_frame_real)
                target_indices = torch.argmax(target_probs, dim=1)
                
                loss_pred_acc += nn.CrossEntropyLoss()(z_next_logits, target_indices)
                current_z_dist = model.sample_from_logits(z_next_logits, temperature=current_temp)
            
            avg_pred_loss = loss_pred_acc / horizon
            
            # Soma simples para validação (não usamos o Tuner aqui para manter a métrica estável)
            loss = loss_rec + avg_pred_loss
            
            total_loss += loss.item()
            steps += 1
            if steps >= max_val_steps: break
            
    return total_loss / max(1, steps)

# ==============================================================================
# 2. LOOP DE TREINO
# ==============================================================================

# CORREÇÃO: Adicionado loss_tuner nos argumentos
def train_one_epoch(model, dataloader, optimizer, device, cfg, current_epoch, temp_scheduler, loss_tuner, scaler):
    model.train()

    total_loss_rec = 0
    total_loss_pred = 0
    total_loss_inv = 0
    steps = 0
    
    start_temp, end_temp, decay_epochs = temp_scheduler
    progress = min(1.0, current_epoch / decay_epochs)
    current_temp = start_temp - (progress * (start_temp - end_temp))
    
    # Previne que a temperatura caia muito rápido (mínimo de 0.5 até a época 30 ajuda)
    current_temp = max(current_temp, 0.1) 

    try:
        total_batches = len(dataloader)
    except:
        total_batches = 100

    pbar = tqdm(dataloader, total=total_batches, desc=f"Ep {current_epoch} (T={current_temp:.2f})", unit="batch")

    use_amp = device.type == 'cuda'
    
    
    for batch_frames, batch_actions in pbar:
        batch_frames = batch_frames.to(device,non_blocking=True)
        batch_actions = batch_actions.to(device,non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            # --- PASSO 0: Estado Inicial ---
            first_frame = batch_frames[:, 0]
            recon, z_dist, _, encoder_logits = model(first_frame, hard=False, temperature=current_temp)
            
            loss_rec = nn.MSELoss()(recon, first_frame)
            
            # --- CÁLCULO DE KL / ENTROPIA (CORRIGIDO: FREE BITS) ---
            # Em vez de explodir o peso, usamos "Free Bits".
            # Comparamos a distribuição do encoder com uma Uniforme (queremos que ele use todos os códigos)
            # N_CAT = 64. Log(64) é a entropia máxima.
            
            # Logits -> LogSoftmax
            log_q = F.log_softmax(encoder_logits, dim=1)
            
            # Prior Uniforme (queremos que ele use tudo)
            # log(1/64) = -log(64)
            log_p = torch.full_like(log_q, -np.log(model.n_categorias))
            
            # KL Divergence: D_KL(Q || P)
            # Queremos minimizar a distância entre o que ele escolhe e a distribuição uniforme
            kl_div = F.kl_div(log_q, log_p.exp(), reduction='batchmean')
            
            # Hinge Loss (Free Bits): Se a KL for pequena (< 1.0), não puna muito.
            # Isso evita o colapso, mas sem a instabilidade do peso dinâmico.
            free_bits = 3.0
            kl_loss = kl_div
            
            loss_pred_acc = 0
            loss_inv_acc = 0
            current_z_dist = z_dist
            
            # --- LOOP MULTISTEP ---
            horizon = batch_actions.shape[1] 
            
            for t in range(horizon):
                action_at_t = batch_actions[:, t]
                next_frame_real = batch_frames[:, t+1]
                
                # 1. Predição
                z_next_logits = model.predict_next_from_dist(current_z_dist, action_at_t)
                
                # 2. Target (Detach para Stop Gradient)
                with torch.no_grad():
                    target_probs = model.get_latent_probs(next_frame_real)
                    target_indices = torch.argmax(target_probs, dim=1)
                
                step_loss_pred = nn.CrossEntropyLoss()(z_next_logits, target_indices)
                loss_pred_acc += step_loss_pred
                
                # 3. Inversa
                with torch.no_grad():
                    _, z_next_dist_real, _, _ = model(next_frame_real, hard=False, temperature=current_temp)
                pred_action_logits = model.predict_action_inverse(current_z_dist, z_next_dist_real)
                step_loss_inv = nn.CrossEntropyLoss()(pred_action_logits, action_at_t.long())
                loss_inv_acc += step_loss_inv
                
                # 4. Próximo passo
                current_z_dist = model.sample_from_logits(z_next_logits, temperature=current_temp, hard=False)

            loss_pred = loss_pred_acc / horizon
            loss_inv = loss_inv_acc / horizon
            
            # --- LOSS TUNER (SEM ENTROPIA) ---
            losses = {
                'rec': loss_rec,
                'pred': loss_pred,
                'inv': loss_inv,
                'ent': torch.tensor(0.0, device=device) # Dummy, o Tuner vai ignorar ou zerar
            }
            
            # O Tuner decide os pesos das tarefas principais
            weighted_loss, current_weights = loss_tuner(losses)
            
            # Adicionamos a KL Loss manualmente com peso FIXO e PEQUENO
            # Isso atua como regularização de fundo, sem brigar com o Tuner
            BETA_KL = 0.01
            total_loss = weighted_loss + (BETA_KL * kl_loss)

        scaler.scale(total_loss).backward()
        
        # Unscale antes de clipar gradiente (opcional, mas bom pra estabilidade)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step otimizado
        scaler.step(optimizer)
        scaler.update()
        
        total_loss_rec += loss_rec.item()
        total_loss_pred += loss_pred.item()
        total_loss_inv += loss_inv.item()
        steps += 1
        
        # Mostra os pesos dinâmicos no log
        pbar.set_postfix({
            'Rec': f"{loss_rec.item():.3f}", 
            'Pred': f"{loss_pred.item():.3f}",
            'KL': f"{kl_loss.item():.3f}",
            'W_Rec': f"{current_weights[0].item():.1f}",
            'W_Inv': f'{current_weights[2].item():.1f}',
            'Inv': f"{loss_inv.item():.3f}",
        })
        
    return total_loss_rec/steps, total_loss_pred/steps, current_temp

# ==============================================================================
# 3. MAIN
# ==============================================================================

def setup_colab_env(dataset_path):
    """
    Se estiver no Colab, move/descompacta os dados para o disco local da VM (/content).
    Ler do Drive montado é MUITO LENTO para treinamento.
    """
    import shutil
    
    # Exemplo: Se o dataset_path for "datasets/carracing_human/train"
    # E você tiver um zip no drive
    print("☁️ Configurando ambiente Colab...")
    
    # Cria pasta local se não existir
    local_path = f"/content/{dataset_path}"
    if not os.path.exists(local_path):
        print(f"📂 Criando diretório local: {local_path}")
        os.makedirs(local_path, exist_ok=True)
        
        # AQUI VOCÊ PODE IMPLEMENTAR LÓGICA DE COPIA/UNZIP SE QUISER
        # Ex: !cp /content/drive/MyDrive/robot/datasets.zip /content/
        # Ex: !unzip /content/datasets.zip
        
        print("⚠️ IMPORTANTE: No Colab, certifique-se que seus dados estão em /content/")
        print("   Ler direto de /content/drive/MyDrive é 10x mais lento.")
    
    return local_path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="minerl_pretrain")
    parser.add_argument("--colab", action="store_true", help="Ativa otimizações para Google Colab")

    args = parser.parse_args()
    
    import sys
    is_running_in_colab = 'google.colab' in sys.modules or args.colab
    
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    if is_running_in_colab:
        print("🚀 MODO COLAB DETECTADO!")
        
    
    
    cfg = load_config(f"configs/{args.config}.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TEMP_START = 2.0
    TEMP_END = 0.1
    TEMP_DECAY_EPOCHS = 300
    temp_scheduler = (TEMP_START, TEMP_END, TEMP_DECAY_EPOCHS)

    exp_name = cfg['training'].get('experiment_name', 'default_run')
    log_dir = os.path.join("logs", exp_name)
    snap_dir = os.path.join(log_dir, "snapshots")
    model_dir = cfg['training']['output_dir']
    
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"📁 Logs: {log_dir}")
    
    

    # Dataset
    max_videos = cfg['dataset'].get('max_videos', None)
    full_dataset_ref = MineRLDataset(cfg['dataset']['train_path'], max_videos=max_videos)
    all_videos = full_dataset_ref.videos
    
    val_count = max(1, int(len(all_videos) * 0.1))
    train_count = len(all_videos) - val_count
    train_videos = all_videos[:train_count]
    val_videos = all_videos[train_count:]
    print(f"📂 Vídeos Treino: {len(train_videos)}, Validação: {len(val_videos)}")
    train_loader = create_dataloader(
        cfg['dataset']['train_path'],
        train_videos, 
        batch_size=cfg['training']['batch_size'], 
        img_size=cfg['dataset'].get('img_size', 64),
        is_colab=is_running_in_colab
    )
    
    val_loader = create_dataloader(
        cfg['dataset']['train_path'],
        val_videos,
        batch_size=cfg['training']['batch_size'], 
        img_size=cfg['dataset'].get('img_size', 64),
        is_colab=is_running_in_colab
    )

    # Modelo
    model = VisualWorldModel(
        input_shape=tuple(cfg['model']['input_shape']),
        n_actions=cfg['model']['n_actions'],
        n_categorias=cfg['model']['n_categories'],
        hidden_size=cfg['model']['hidden_size'],
        cnn_channels=cfg['model'].get('cnn_channels', 32)
    ).to(device)
    
    if is_running_in_colab and torch.__version__ >= "2.0.0":
        print("🔥 Ativando torch.compile()... (Primeira época será lenta, depois voa)")
        model = torch.compile(model)
    
    # Loss Tuner
    loss_tuner = DynamicLossTuner(n_losses=4).to(device)
    
    # Otimizador (Inclui parametros do modelo E do tuner)
    optimizer = optim.Adam(
        list(model.parameters()) + list(loss_tuner.parameters()), 
        lr=cfg['training']['learning_rate']
    )

    # Checkpoint
    save_path_best = os.path.join(model_dir, "best_visual_model.pth")
    save_path_last = os.path.join(model_dir, "last_visual_model.pth")
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(save_path_last):
        print(f"🔄 Carregando: {save_path_last}")
        checkpoint = torch.load(save_path_last, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Tenta carregar estado do tuner se existir (se for checkpoint antigo, ignora)
        if 'loss_tuner_state_dict' in checkpoint:
            loss_tuner.load_state_dict(checkpoint['loss_tuner_state_dict'])
            
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"🚀 Iniciando treino...")
    patience = cfg['training'].get('patience', 20) # Lê do config, padrão 20
    epochs_no_improve = 0
    print(f"🛑 Early Stopping ativado: Paciência de {patience} épocas.")
    try:
        for epoch in range(start_epoch, cfg['training']['epochs']):
            start_time = time.time()
            
            # CORREÇÃO: Passando loss_tuner
            avg_rec, avg_pred, curr_temp = train_one_epoch(
                model, train_loader, optimizer, device, 
                cfg, epoch, temp_scheduler, loss_tuner,
                scaler=scaler
            )
            
            val_loss = validate(model, val_loader, device, cfg, curr_temp)
            save_snapshot(model, val_loader, device, epoch, snap_dir, curr_temp)
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path_best)
                epochs_no_improve = 0 # Zeramos o contador pq melhorou!
                status = "🏆 BEST"
            else:
                epochs_no_improve += 1
                status = f"⏳ ({epochs_no_improve}/{patience})"
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_tuner_state_dict': loss_tuner.state_dict(), # Salva estado do tuner
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, save_path_last)
            
            print(f"✅ Ep {epoch} | Val: {val_loss:.4f} | Rec: {avg_rec:.3f} | Pred: {avg_pred:.3f} | {status}")
            
            if epochs_no_improve >= patience:
                print(f"\n🛑 EARLY STOPPING ACIONADO!")
                print(f"   O modelo não melhora há {patience} épocas.")
                print(f"   Melhor Val Loss foi: {best_val_loss:.4f}")
                break # Sai do loop for
            
    except KeyboardInterrupt:
        print("\n🛑 Treino pausado.")

if __name__ == "__main__":
    main()