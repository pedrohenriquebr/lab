import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from esp32robot.agents import DiscreteWorldModel
from esp32robot.simulation import Esp322DEnv

# ==============================================================================
# 1. A ARQUITETURA DO MODELO DE MUNDO (Discrete/Gumbel)
# ==============================================================================


# ==============================================================================
# 2. COLETA DE DADOS (Seu Simulador)
# ==============================================================================
def collect_data(steps=2000):
    print("🤖 Coletando dados do Esp322DEnv...")
    env = Esp322DEnv(render_mode='human', env_type='maze', stack_size=4, latency_steps=5)
    obs, _ = env.reset()
    
    buffer = []
    
    for _ in range(steps):
        action = env.action_space.sample() # Ação aleatória serve para aprender física
        # action  = (action % 4) + 1
        next_obs, reward, done, _, _ = env.step(action)
        
        # Guarda (Estado, Ação, Próximo Estado)
        buffer.append((obs, action, next_obs))
        
        obs = next_obs
        if done: obs, _ = env.reset()
        
    env.close()
    return buffer

# ==============================================================================
# 3. TREINAMENTO E VALIDAÇÃO
# ==============================================================================
def train_world_model():
    # Configurações
    INPUT_DIM = 12  # 3 sensores * 4 frames (confirme seu stack_size)
    CATEGORIAS = 32 # Teste com 16 ou 32
    EPOCHS = 300
    BATCH_SIZE = 64
    
    # Prepara dados
    raw_data = collect_data(steps=6000)
    
    # Converte para Tensores
    states = torch.tensor(np.array([x[0] for x in raw_data]), dtype=torch.float32)
    actions = torch.tensor(np.array([x[1] for x in raw_data]), dtype=torch.long)
    next_states = torch.tensor(np.array([x[2] for x in raw_data]), dtype=torch.float32)
    
    # Instancia Modelo
    wm = DiscreteWorldModel(input_dim=INPUT_DIM, n_categorias=CATEGORIAS)
    optimizer = optim.Adam(wm.parameters(), lr=0.001)
    
    rec_losses = []
    pred_losses = []

    print(f"\n🧠 Treinando World Model ({CATEGORIAS} categorias latentes)...")
    
    for epoch in range(EPOCHS):
        # Embaralha índices
        indices = torch.randperm(len(states))
        
        epoch_rec_loss = 0
        epoch_pred_loss = 0
        
        for i in range(0, len(states), BATCH_SIZE):
            idx = indices[i:i+BATCH_SIZE]
            batch_s = states[idx]
            batch_a = actions[idx]
            batch_ns = next_states[idx]
            
            # --- FORWARD ---
            # hard=False no começo ajuda a aprender, mas hard=True é o objetivo final
            # Vamos usar hard=True direto para forçar lógica discreta
            rec_s, z_dist, z_future_logits, _ = wm(batch_s, batch_a, hard=False)
            
            # --- GABARITO DO FUTURO ---
            # Precisamos saber qual categoria o encoder escolhe para o next_state REAL
            with torch.no_grad():
                # Passamos o next_state pelo encoder para saber o "Label" correto
                target_logits = wm.encoder(batch_ns)
                target_class = torch.argmax(target_logits, dim=1) # O índice correto (0 a 31)

            # --- CÁLCULO DAS PERDAS ---
            # 1. Loss de Reconstrução (O Decoder entendeu o Encoder?)
            loss_rec = F.mse_loss(rec_s, batch_s)
            
            # 2. Loss de Previsão (O Predictor acertou a classe do futuro?)
            loss_pred = F.cross_entropy(z_future_logits, target_class)
            
            # Soma ponderada (geralmente reconstrução é mais fácil, damos peso pro preditor)
            loss = loss_rec + (0.5 * loss_pred)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_rec_loss += loss_rec.item()
            epoch_pred_loss += loss_pred.item()
            
        # Logs
        if epoch % 20 == 0:
            avg_rec = epoch_rec_loss / (len(states)/BATCH_SIZE)
            avg_pred = epoch_pred_loss / (len(states)/BATCH_SIZE)
            rec_losses.append(avg_rec)
            pred_losses.append(avg_pred)
            print(f"Ep {epoch}: Rec Loss={avg_rec:.4f} | Pred Loss={avg_pred:.4f}")

    # ==========================================================================
    # 4. VISUALIZAÇÃO DA "VIABILIDADE"
    # ==========================================================================
    wm.eval()
    
    print("\n📊 Gerando gráficos de validação...")
    
    # Teste em um único exemplo
    idx_test = 100
    test_state = states[idx_test].unsqueeze(0)
    test_action = actions[idx_test].unsqueeze(0)
    
    with torch.no_grad():
        rec, z_dist, z_fut_logits, _ = wm(test_state, test_action, hard=True)
        cat_idx = torch.argmax(z_dist).item()
        pred_idx = torch.argmax(z_fut_logits).item()
        
        # Gabarito real do futuro
        next_real = next_states[idx_test].unsqueeze(0)
        logits_next = wm.encoder(next_real)
        real_next_idx = torch.argmax(logits_next).item()

    print(f"\n--- Teste Unitário (Amostra #{idx_test}) ---")
    print(f"Entrada (12 sensores): {test_state.numpy().round(2)}")
    print(f"Reconstrução:          {rec.numpy().round(2)}")
    print(f"Categoria Latente:     {cat_idx}")
    print(f"Ação Tomada:           {test_action.item()}")
    print(f"Previsto (Futuro):     Categoria {pred_idx}")
    print(f"Real (Futuro):         Categoria {real_next_idx}")
    
    if pred_idx == real_next_idx:
        print("✅ O MODELO PREVIU O FUTURO CORRETAMENTE!")
    else:
        print("❌ Previsão incorreta (normal no começo/com poucos dados).")

    # Plotar comparação visual (Sensores Reais vs Reconstruídos)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(test_state[0].numpy(), label='Original', color='blue', marker='o')
    plt.plot(rec[0].numpy(), label='Reconstruído (Sonho)', color='red', linestyle='--', marker='x')
    plt.title(f"Reconstrução (Latente #{cat_idx})")
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(rec_losses, label='Loss Reconstrução')
    plt.plot(pred_losses, label='Loss Previsão')
    plt.title("Curvas de Aprendizado")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("world_model_test.png")
    print("📸 Gráfico salvo em 'world_model_test.png'")
    plt.close()
    
    torch.save(wm.state_dict(), "mini_world_model.pth")
    print("💾 Modelo salvo como mini_world_model.pth")

if __name__ == "__main__":
    train_world_model()