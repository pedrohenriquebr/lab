import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os

from esp32robot.utils import load_config
from esp32robot.agents_text import LangJEPA
from esp32robot.data_text import create_text_dataloader

def visualize(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carrega dados e modelo
    dataloader, tokenizer = create_text_dataloader(cfg, shuffle=False)
    
    model = LangJEPA(
        vocab_size=tokenizer.vocab_count,
        embed_dim=cfg['model']['embed_dim'],
        hidden_size=cfg['model']['hidden_size']
    ).to(device)
    
    model_path = os.path.join(cfg['training']['output_dir'], "last_jepa.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Pesos carregados.")
    else:
        print("⚠️ Pesos não encontrados, rodando aleatório.")
        
    model.eval()
    
    # Pega uma amostra
    context, future = next(iter(dataloader))
    context = context[0:1].to(device) # Pega só o primeiro exemplo
    future = future[0:1].to(device)
    
    # Texto legível
    ctx_text = tokenizer.decode(context[0].cpu().tolist())
    fut_text = tokenizer.decode(future[0].cpu().tolist())
    
    print(f"\n📝 Contexto: '{ctx_text}'")
    print(f"🔮 Futuro Real: '{fut_text}'")
    
    with torch.no_grad():
        loss, z_pred, z_target = model(context, future)
        
        # Similaridade de Cosseno (1.0 = Perfeito, 0.0 = Nada a ver, -1.0 = Oposto)
        similarity = F.cosine_similarity(z_pred, z_target, dim=1).item()
        
        # Distância Euclidiana (Quanto menor melhor)
        dist = torch.dist(z_pred, z_target).item()

    print(f"\n📊 Análise Semântica:")
    print(f"   Distância Vetorial (MSE Loss): {loss.item():.6f}")
    print(f"   Similaridade de Significado: {similarity:.4f}")
    
    # Gráfico dos Vetores
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Pensamento Previsto (Pred)")
    plt.imshow(z_pred.cpu().numpy().reshape(16, -1), aspect='auto', cmap='viridis')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Pensamento Real (Target)")
    plt.imshow(z_target.cpu().numpy().reshape(16, -1), aspect='auto', cmap='viridis')
    plt.axis('off')
    
    plt.suptitle(f"Comparação Mental (Sim: {similarity:.2f})")
    plt.tight_layout()
    plt.savefig("jepa_thought_viz.png")
    print("📸 Gráfico salvo em 'jepa_thought_viz.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="llm_jepa")
    args = parser.parse_args()
    
    cfg = load_config(f"configs/{args.config}.yaml")
    visualize(cfg)