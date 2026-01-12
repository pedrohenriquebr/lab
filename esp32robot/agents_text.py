import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LangJEPA(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_size=512, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # --- 1. COMPONENTES COMPARTILHADOS ---
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(embed_dim)
        
        # --- 2. ENCODER (Contexto / Online) ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_size, dropout=dropout, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- 3. TARGET ENCODER (Futuro / EMA) ---
        # Cópia exata do encoder, mas não treina via gradiente
        self.target_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Inicializa com os mesmos pesos
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        # Desliga gradientes do Target (ele aprende via EMA)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # --- 4. PREDICTOR (O Cérebro JEPA) ---
        # Tenta transformar: (Resumo do Contexto) -> (Resumo do Futuro)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_dim) # Saída deve bater com o target_encoder
        )

    def encode_context(self, x):
        # x: [Batch, Seq_Len]
        x = self.token_emb(x)
        x = self.pos_emb(x)
        # Retorna a sequência completa de embeddings contextuais
        return self.context_encoder(x)

    def encode_target(self, x):
        # Usado apenas para gerar o alvo (Ground Truth)
        with torch.no_grad():
            x = self.token_emb(x)
            x = self.pos_emb(x)
            return self.target_encoder(x)

    def forward(self, context_ids, future_ids):
        """
        JEPA Forward Pass:
        1. Contexto -> Z_Online
        2. Futuro -> Z_Target (Sem Gradiente)
        3. Predictor(Z_Online) -> Z_Pred
        4. Loss = MSE(Z_Pred, Z_Target)
        """
        
        # 1. Processa Contexto (Rede que aprende)
        z_ctx_seq = self.encode_context(context_ids)
        # Pegamos o vetor do ÚLTIMO token como resumo do contexto
        z_ctx_summary = z_ctx_seq[:, -1, :] 
        
        # 2. Predição (Imagine o significado do futuro)
        z_pred = self.predictor(z_ctx_summary)
        
        # 3. Processa Futuro (Rede Estável - Target)
        z_tgt_seq = self.encode_target(future_ids)
        # Queremos prever o significado global do bloco futuro.
        # Podemos usar a média dos vetores futuros ou o último vetor.
        # Vamos usar a média para capturar o "tópico" geral.
        z_tgt_summary = z_tgt_seq.mean(dim=1) 
        
        # 4. Cálculo da Loss Vetorial (Feature Space)
        # Não estamos prevendo palavras! Estamos prevendo vetores.
        loss = F.mse_loss(z_pred, z_tgt_summary)
        
        return loss, z_pred, z_tgt_summary

    def update_target_ema(self, decay):
        """Atualiza os pesos do Target Encoder suavemente"""
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * decay + param_q.data * (1. - decay)