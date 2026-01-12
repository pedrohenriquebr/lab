import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LangJEPA(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=256, hidden_size=512, n_layers=4):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # --- 1. COMPONENTES ---
        # Embedding (Token -> Vetor)
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(embed_dim, max_len=1000)
        
        # Encoder (Contexto - Rede Online)
        # É quem "lê" o texto e cria o pensamento atual
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Target Encoder (Futuro - Rede Alvo)
        # É uma cópia do Encoder que não treina via gradiente, só via EMA
        self.target_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Inicializa pesos iguais e desliga gradiente do Target
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor (O Cérebro JEPA)
        # Tenta transformar (Contexto) -> (Representação do Futuro)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_dim) # Tenta bater com o target_encoder
        )
        
        # (Opcional) Cabeça de Texto para Debug
        # Só pra gente ver se ele está gerando palavras reais, mas o treino JEPA não usa isso na Loss principal
        self.vocab_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x_context, x_future=None):
        """
        x_context: Tokens do passado [Batch, Seq_Len]
        x_future: Tokens do futuro que queremos prever (Target)
        """
        # 1. Codifica Contexto (Online)
        emb_ctx = self.token_emb(x_context)
        emb_ctx = self.pos_emb(emb_ctx)
        
        # Z_context: [Batch, Seq_Len, Dim]
        z_context = self.context_encoder(emb_ctx)
        
        # Pega apenas o último estado (resumo do passado)
        z_last = z_context[:, -1, :] 
        
        # 2. Predição (Imagine o significado do futuro)
        # Aqui o modelo tenta adivinhar qual será o vetor do target
        z_pred = self.predictor(z_last) 
        
        loss = None
        
        if x_future is not None:
            # 3. Codifica Futuro (Target) - SEM GRADIENTE
            with torch.no_grad():
                emb_fut = self.token_emb(x_future)
                emb_fut = self.pos_emb(emb_fut)
                z_target_seq = self.target_encoder(emb_fut)
                
                # Queremos prever o vetor do primeiro token do futuro (ou média)
                z_target = z_target_seq[:, 0, :] 
            
            # 4. JEPA LOSS (Distância no Espaço Latente)
            # Não usamos CrossEntropy! Usamos MSE ou Cosine.
            # O modelo aprende a aproximar os vetores, não as palavras.
            loss = F.mse_loss(z_pred, z_target)
            
            # (Regularização opcional) Forçar z a não ser zero
            # loss += 0.01 * (torch.mean(z_pred**2) - 1.0)**2 

        return z_pred, loss

    def update_target_ema(self, decay=0.99):
        """Atualiza o Target Encoder com média móvel do Context Encoder"""
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * decay + param_q.data * (1. - decay)

    def generate_text(self, z):
        """Debug: Tenta converter o pensamento (z) em palavra"""
        logits = self.vocab_head(z)
        token_id = torch.argmax(logits, dim=-1)
        return token_id

# Helper de Posição (Padrão Transformer)
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