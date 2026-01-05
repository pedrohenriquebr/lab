import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from esp32robot.agents import DiscreteWorldModel
from esp32robot.simulation import Esp322DEnv
import time

# ==============================================================================
# 2. CONFIGURAÇÃO DA VISUALIZAÇÃO
# ==============================================================================
def run_live_inference():
    # Carregar Modelo
    model = DiscreteWorldModel(input_dim=12, n_categorias=32) # Ajuste se usou 32
    try:
        model.load_state_dict(torch.load("mini_world_model.pth"))
        print("✅ Modelo carregado com sucesso!")
    except:
        print("❌ Erro: Arquivo 'mini_world_model.pth' não encontrado. Treine primeiro!")
        return
    
    model.eval()

    # Iniciar Ambiente (Modo Human para ver o PyGame)
    env = Esp322DEnv(render_mode="human", env_type='default', stack_size=4,latency_steps=5)
    obs, _ = env.reset()

    # Configurar Matplotlib Interativo
    plt.ion() 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    # Linhas do gráfico
    line_real, = ax1.plot(np.zeros(12), label='Real (Input)', color='blue', linewidth=2)
    line_dream, = ax1.plot(np.zeros(12), label='Sonho (Reconstrução)', color='red', linestyle='--', linewidth=2)
    
    # Barras de probabilidade (Qual estado ele acha que está?)
    bar_probs = ax2.bar(range(model.n_categorias), np.zeros(model.n_categorias), color='purple')

    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title("O que o Robô Vê vs. O que ele Imagina")
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Estado Latente Ativo (O 'Código' do Momento)")
    ax2.set_xlabel("ID da Categoria")

    print("\n🚀 Iniciando loop de inferência... (Feche a janela do gráfico para sair)")

    for step in range(1000):
        # 1. Agente burro (aleatório) ou manual, só pra gerar dados
        action = env.action_space.sample()
        
        # 2. Passar pelo Modelo de Mundo
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long)
        model.eval()
        
        with torch.no_grad():
            # Pegamos a reconstrução e os logits (probabilidades cruas)
            rec, z_dist, z_fut_logits, logits = model(obs_tensor, action_tensor, hard=True)
            
            # Para visualizar melhor as barras, pegamos os logits do encoder puros
            logits_encoder = model.encoder(obs_tensor)
            probs = F.softmax(logits_encoder, dim=1).numpy()[0]
            
            # Previsão do futuro (Physics)
            fut_cat = torch.argmax(z_fut_logits).item()

        # 3. Atualizar Gráficos
        # Linhas
        line_real.set_ydata(obs)
        line_dream.set_ydata(rec.numpy()[0])
        
        # Barras (Qual estado acendeu?)
        for rect, h in zip(bar_probs, probs):
            rect.set_height(h)
            # Pinta de vermelho se for a categoria ativa
            rect.set_color('red' if h > 0.5 else 'purple')

        ax1.set_title(f"Step {step} | Ação: {action} | Prevê Futuro: Cat #{fut_cat}")
        
        fig.canvas.draw()
        fig.canvas.flush_events()

        # 4. Passo no Real
        next_obs, reward, done, _, _ = env.step(action)
        obs = next_obs
        
        if done:
            obs, _ = env.reset()
            print("🔄 Resetando ambiente...")

        # Pausa para dar tempo de ver (sincronizar PyGame e Matplotlib)
        plt.pause(0.05) 

    plt.ioff()
    plt.show()
    env.close()

if __name__ == "__main__":
    run_live_inference()