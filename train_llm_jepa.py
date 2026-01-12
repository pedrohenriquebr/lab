import torch
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Imports locais
from esp32robot.utils import load_config
from esp32robot.agents_text import LangJEPA
from esp32robot.data_text import create_text_dataloader

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando treino Lang-JEPA no device: {device}")

    # 1. Dados
    dataloader, tokenizer = create_text_dataloader(cfg)
    
    # 2. Modelo
    model = LangJEPA(
        vocab_size=tokenizer.vocab_count, # Ajusta ao real
        embed_dim=cfg['model']['embed_dim'],
        hidden_size=cfg['model']['hidden_size'],
        n_layers=cfg['model']['n_layers'],
        n_heads=cfg['model']['n_heads'],
        dropout=cfg['model']['dropout']
    ).to(device)
    
    # Otimizador (AdamW é essencial para Transformers)
    optimizer = optim.AdamW(
        model.predictor.parameters(), # No JEPA puro, focamos no predictor e encoder online
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    # Adicionamos parametros do encoder online no otimizador
    optimizer.add_param_group({'params': model.context_encoder.parameters()})
    
    # 3. Loop
    ema_decay = cfg['training']['ema_decay']
    output_dir = cfg['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    loss_history = []

    model.train()
    print("🧠 Começando a aprender representações semânticas...")
    
    try:
        for epoch in range(cfg['training']['epochs']):
            pbar = tqdm(dataloader, desc=f"Ep {epoch}")
            epoch_loss = 0
            
            for context, future in pbar:
                context, future = context.to(device), future.to(device)
                
                # --- JEPA STEP ---
                loss, z_pred, z_target = model(context, future)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Update Target Network (EMA)
                model.update_target_ema(ema_decay)
                
                epoch_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.6f}"})
            
            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)
            print(f"✅ Ep {epoch} Final Loss: {avg_loss:.6f}")
            
            # Save Checkpoint
            torch.save(model.state_dict(), os.path.join(output_dir, "last_jepa.pth"))

    except KeyboardInterrupt:
        print("\n🛑 Treino pausado.")
    
    print("💾 Modelo salvo.")
    return loss_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="llm_jepa")
    args = parser.parse_args()
    
    cfg = load_config(f"configs/{args.config}.yaml")
    train(cfg)