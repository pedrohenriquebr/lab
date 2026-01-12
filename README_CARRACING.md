# 🏎️ Visual World Model - CarRacing V3

Documentação da configuração otimizada para treinar um modelo de mundo visual no ambiente `CarRacing-v2`.

## 📋 Visão Geral
Este modelo aprende a:
1. **Ver:** Comprimir frames do jogo (96x96) em um vetor latente discreto.
2. **Imaginar:** Prever o próximo frame baseado na ação tomada.
3. **Entender:** Deduzir qual ação causou uma mudança visual.

## ⚙️ Configuração Vencedora (`carracing_pretrain.yaml`)

Esta configuração foi ajustada para evitar *Codebook Collapse* e garantir nitidez visual.

```yaml
dataset:
  train_path: "datasets/carracing_human/train" # Use dados gravados por humanos!
  test_path: "datasets/carracing_human/test"
  img_size: 96          # Resolução nativa do Gym
  action_mapping: true  # Usa o mapeamento discreto (0-4)
  # max_videos: 5       # COMENTAR para usar dataset completo

model:
  input_shape: [3, 96, 96]
  n_actions: 5          # Stop, Gas, Brake, Left, Right
  
  # --- Arquitetura ---
  n_categories: 256     # Vocabulário visual rico (evita borrões)
  hidden_size: 256      # Cérebro de física equilibrado
  cnn_channels: 48      # "Olhos de Águia" (mais detalhes que o padrão 32)

training:
  experiment_name: "carracing_v3_human"
  epochs: 1000
  patience: 100
  batch_size: 64
  learning_rate: 0.001
  
  # --- Pesos da Loss (Controlados pelo DynamicTuner) ---
  # O código usa DynamicLossTuner, então esses valores iniciais
  # servem apenas de referência, o modelo ajusta sozinho.
  beta_rec: 1.0
  beta_pred: 1.0
  beta_inverse: 1.0