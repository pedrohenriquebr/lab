import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        # Caracteres reservados
        self.pad_token = 0
        self.unk_token = 1
        self.vocab_count = 2

    def fit_on_text(self, text):
        # Tokenização simples por caractere para robustez
        unique_chars = sorted(list(set(text)))
        for char in unique_chars:
            if self.vocab_count < self.vocab_size:
                self.char_to_idx[char] = self.vocab_count
                self.idx_to_char[self.vocab_count] = char
                self.vocab_count += 1
        print(f"🔤 Tokenizer treinado. Vocab size: {self.vocab_count}")

    def encode(self, text):
        return [self.char_to_idx.get(c, self.unk_token) for c in text]

    def decode(self, indices):
        return "".join([self.idx_to_char.get(i, "?") for i in indices])

class TextJepaDataset(Dataset):
    def __init__(self, file_path, seq_len=64, pred_len=16, vocab_size=10000):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Garante que o arquivo existe (cria dummy se não existir)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            print(f"⚠️ Arquivo {file_path} não encontrado. Criando dummy text...")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("O rato roeu a roupa do rei de roma. " * 1000)
        
        # Carrega texto
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()
            
        # Tokeniza
        self.tokenizer = SimpleTokenizer(vocab_size)
        self.tokenizer.fit_on_text(self.text)
        self.data = torch.tensor(self.tokenizer.encode(self.text), dtype=torch.long)
        
        # Total de amostras possíveis
        self.total_len = len(self.data) - (seq_len + pred_len)
        print(f"📚 Dataset carregado: {len(self.data)} tokens.")

    def __len__(self):
        return max(0, self.total_len)

    def __getitem__(self, idx):
        # Pega janela deslizante
        # Contexto: [idx ... idx + seq_len]
        # Futuro:   [idx + seq_len ... idx + seq_len + pred_len]
        
        ctx_end = idx + self.seq_len
        fut_end = ctx_end + self.pred_len
        
        context = self.data[idx : ctx_end]
        future = self.data[ctx_end : fut_end]
        
        return context, future

def create_text_dataloader(config, shuffle=True):
    dataset = TextJepaDataset(
        config['dataset']['train_path'],
        seq_len=config['dataset']['seq_len'],
        pred_len=config['dataset']['pred_len'],
        vocab_size=config['model']['vocab_size']
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=0, # Seguro pra windows
        pin_memory=True
    )
    return loader, dataset.tokenizer