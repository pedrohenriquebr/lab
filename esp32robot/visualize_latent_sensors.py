import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Importações do Projeto
from esp32robot.simulation import Esp322DEnv
from esp32robot.agents import DiscreteWorldModel
from esp32robot.utils import load_config, ConfigurationParameters
from esp32robot.mappers import map_config_params_to_dict

# Configurações Padrão (Fallback)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_latent_data(env, model, n_samples, latent_mode='logits'):
    print(f"🤖 Coletando {n_samples} amostras do ambiente...")
    
    latents = []
    errors = []
    sensor_states = [] 
    
    obs, _ = env.reset()
    
    for i in range(n_samples):
        # Prepara entrada: (12,) -> (1, 12)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # forward: rec, z_dist, future, logits
            recon, z_dist, _, logits = model(obs_tensor)
            
            # Escolhemos o vetor latente
            if latent_mode == 'logits':
                vec = logits.cpu().numpy()[0]
            else:
                vec = z_dist.cpu().numpy()[0]
            
            # Calcula Erro de Reconstrução
            loss = nn.MSELoss()(recon, obs_tensor)
            
        latents.append(vec)
        errors.append(loss.item())
        
        # Pega a média dos sensores atuais (últimos 3 do stack) para colorir o gráfico
        current_sensors = obs[-env.num_sensors:] 
        avg_dist = np.mean(current_sensors)
        sensor_states.append(avg_dist)
        
        # Ação aleatória para explorar
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        
        if done: obs, _ = env.reset()
            
        if i % 100 == 0:
            print(f"   Coletado: {i}/{n_samples}", end='\r')
            
    return np.array(latents), np.array(errors), np.array(sensor_states)

def visualize_3d(data_2d, colors, title, label_name):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    p = ax.scatter(data_2d[:, 0], data_2d[:, 1], data_2d[:, 2], 
                   c=colors, cmap='viridis', alpha=0.6, s=15)
    
    cbar = fig.colorbar(p, label=label_name, pad=0.1)
    ax.set_title(title)
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.set_zlabel('Componente 3')
    plt.show()
    
    
def visualize_2d(data_2d, colors, title, label_name):
    plt.figure(figsize=(10, 8))
    p = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                    c=colors, cmap='viridis', alpha=0.6, s=15)
    cbar = plt.colorbar(p, label=label_name)
    plt.title(title)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualizador 3D do Espaço Latente (Sensores)")
    parser.add_argument("--config", type=str, required=True, help="Nome do arquivo de config (sem .yaml) na pasta configs/")
    parser.add_argument("--samples", type=int, default=5000, help="Número de amostras para coletar")
    parser.add_argument("--method", type=str, default='PCA', choices=['PCA', 'TSNE'], help="Método de redução (PCA ou TSNE)")
    parser.add_argument("--mode", type=str, default='logits', choices=['logits', 'probs'], help="Usar logits brutos ou probabilidades softmax")
    
    args = parser.parse_args()
    
    # 1. Carrega Configurações
    config_path = os.path.join('./configs/', args.config + '.yaml')
    print(f"📄 Lendo configuração: {config_path}")
    
    config_data = load_config(config_path)
    
    # Mapeia para dicionário plano (flat) igual ao evaluate.py
    cfg_obj = ConfigurationParameters(
        environment_config=config_data.get("environment", {}),
        agent_config=config_data.get("agent", {}),
        world_model_config=config_data.get("world_model", {}),
        training_config=config_data.get("training", {})
    )
    params = map_config_params_to_dict(cfg_obj)

    # 2. Inicializa Ambiente (Usando params do config)
    print("🌍 Inicializando Ambiente...")
    env = Esp322DEnv(
        render_mode=None, # Headless para ser rápido
        env_type=params.get('env_type', 'maze'),
        stack_size=params.get('stack_size', 4),
        latency_steps=params.get('latency_steps', 0),
        # Passa outros params se necessário (thresholds, etc.)
    )
    
    # 3. Inicializa e Carrega World Model
    input_dim = env.observation_space.shape[0]
    n_cats = params.get('world_model_n_categories', 32)
    hidden = params.get('world_model_hidden_size', 64)
    model_path = params.get('world_model_path', 'mini_world_model.pth')
    
    print(f"🧠 Inicializando World Model: Input={input_dim}, Cats={n_cats}, Hidden={hidden}")
    model = DiscreteWorldModel(input_dim=input_dim, n_categorias=n_cats, hidden_size=hidden).to(DEVICE)
    
    # Tenta carregar pesos
    # Verifica se o path é absoluto ou relativo
    if not os.path.exists(model_path):
        # Tenta procurar na pasta models/ ou output_dir
        alt_path = os.path.join(params.get('output_dir', ''), model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
            
    try:
        print(f"📂 Carregando pesos de: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("✅ Pesos carregados!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print("⚠️ Rodando com pesos aleatórios (Visualização será apenas estrutural, sem significado aprendido)")

    model.eval()
    
    # 4. Coleta e Processamento
    X, errors, sensors = collect_latent_data(env, model, args.samples, args.mode)
    
    print(f"\n📐 Calculando {args.method} em 3D para {len(X)} pontos...")
    if args.method == 'PCA':
        reducer = PCA(n_components=2)
    else:
        # Perplexity menor ajuda se tiver poucos dados
        reducer = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
        
    X_embedded = reducer.fit_transform(X)
    
    # 5. Visualização
    print("📊 Plot 1: Mapa Mental (Baseado em Distância do Sensor)")
    visualize_2d(X_embedded, sensors, 
                 f"Espaço Latente ({args.method}) - Organização do Conhecimento\nCor: Distância Média (Amarelo=Livre, Roxo=Perto)", 
                 "Sensor (0=Perto, 1=Longe)")

    print("📊 Plot 2: Mapa de Confiança (Baseado em Erro de Reconstrução)")
    visualize_2d(X_embedded, errors, 
                 f"Espaço Latente ({args.method}) - Zonas de Dificuldade\nCor: Erro MSE (Roxo=Baixo/Bom, Amarelo=Alto/Confuso)", 
                 "Erro de Reconstrução")

if __name__ == "__main__":
    main()