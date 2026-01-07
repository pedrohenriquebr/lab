import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

class MineRLDataset(IterableDataset):
    def __init__(self, root_path, img_size=64, action_mapper=None, max_videos=None, videos=None):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), # Muda luz
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0),
        ])
        self.root_path = root_path
        self.img_size = img_size
        all_videos = [f for f in os.listdir(root_path) if f.endswith('.mp4')] if videos is None else videos
        if max_videos is not None:
            self.videos = all_videos[:max_videos]
            print(f"✂️ Dataset cortado: Usando apenas {len(self.videos)} vídeos (de {len(all_videos)} disponíveis).")
        else:
            self.videos = all_videos
        self.action_mapper = action_mapper if action_mapper else self._default_mapper
        
        # Carrega metadados uma vez
        self.metadata = {}
        meta_path = os.path.join(root_path, "metadata.json")
        if os.path.exists(meta_path):
            print(f"📂 Carregando metadados de {meta_path}...")
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        
        # --- NOVO: Pré-cálculo para o ETA funcionar ---
        self.total_samples = 0
        print(f"📊 Indexando {len(self.videos)} vídeos para estimar tempo...")
        
        for vid_name in self.videos:
            vid_path = os.path.join(self.root_path, vid_name)
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened(): continue
            
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Ajusta baseado no limite do metadata se existir
            meta_entry = self.metadata.get(vid_name, None)
            if meta_entry and 'forward' in meta_entry:
                n_frames = min(n_frames, len(meta_entry['forward']))
            
            # Cada frame gera 1 amostra (exceto o último que não tem 'next')
            if n_frames > 1:
                self.total_samples += (n_frames - 1)
            cap.release()
            
        print(f"✅ Total de frames para treino: {self.total_samples}")

    def __len__(self):
        """Permite que o tqdm saiba o total"""
        return self.total_samples

    def _default_mapper(self, meta_entry, frame_idx):
            """
            Mapeia combinações de teclas e mouse para classes discretas (0-13).
            Prioridade: Ações Raras > Movimento Combinado > Movimento Simples
            """
            if meta_entry is None: return 0

            # 1. Extração de Dados Brutos (com tratamento de erro)
            # Mouse (Camera)
            cam_x = meta_entry.get('camera_x', [0]*10000)[frame_idx]
            cam_y = meta_entry.get('camera_y', [0]*10000)[frame_idx]
            
            # Teclas (0 ou 1)
            fwd = meta_entry.get('forward', [0])[frame_idx]
            back = meta_entry.get('back', [0])[frame_idx]
            jump = meta_entry.get('jump', [0])[frame_idx]
            attack = meta_entry.get('attack', [0])[frame_idx]
            place = meta_entry.get('place', [0])[frame_idx]
            
            # 2. Definição de Limiares (Baseado no seu JSON)
            # No JSON, 0.15 parece ruído leve, 0.45+ já é movimento intencional.
            CAM_THRESHOLD = 0.4 
            CAM_Y_THRESHOLD = 2.0 # Olhar muito pra cima ou baixo

            # Detecta Giros
            turn_left = cam_x < -CAM_THRESHOLD
            turn_right = cam_x > CAM_THRESHOLD
            
            # Detecta Pitch (Opcional, mas bom pra saber se está construindo ponte)
            look_up = cam_y < -CAM_Y_THRESHOLD
            look_down = cam_y > CAM_Y_THRESHOLD

            # --- LÓGICA DE DECISÃO (Hierárquica) ---

            # Prioridade 1: Interações Específicas
            if place: return 11
            
            # Se estiver atacando...
            if attack:
                if fwd: return 10  # Minerar andando
                return 9           # Minerar parado

            # Prioridade 2: Movimento + Pulo (Parkour)
            if jump:
                if fwd: return 2   # Pulo p/ frente
                return 8           # Pulo vertical

            # Prioridade 3: Movimento + Curva (Direção do olhar domina)
            if fwd:
                if turn_left: return 3  # Curva Esq
                if turn_right: return 4 # Curva Dir
                return 1                # Só frente

            # Prioridade 4: Movimento Simples
            if back: return 5

            # Prioridade 5: Apenas Câmera (Girar no eixo)
            if turn_left: return 6
            if turn_right: return 7
            
            # Prioridade 6: Apenas Pitch (Olhar pro chão/céu sem andar)
            if look_up: return 12
            if look_down: return 13

            # Se nada aconteceu
            return 0 # IDLE
        
    def _process_frame(self, frame):
            # 1. Resize (Mantém OpenCV porque é rápido)
            resized = cv2.resize(frame, (self.img_size, self.img_size))
            
            # 2. Garante RGB (OpenCV lê em BGR)
            # O ColorJitter precisa das cores certas para alterar saturação/hue
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # 3. Aplica o Augmentation (Luz/Cor) + Conversão pra Tensor
            if hasattr(self, 'transform') and self.transform:
                # O transform já faz: ToPIL -> Jitter -> ToTensor (Div 255 + Permute)
                return self.transform(rgb)
            
            # Fallback (Caso você tire o transform no futuro, mantém o manual)
            tensor = torch.from_numpy(rgb).float() / 255.0
            return tensor.permute(2, 0, 1)

    def _video_generator(self):
        """Generator que itera vídeo por vídeo, frame por frame"""
        for vid_name in self.videos:
            vid_path = os.path.join(self.root_path, vid_name)
            cap = cv2.VideoCapture(vid_path)
            
            meta_entry = self.metadata.get(vid_name, {})
            # Fallback seguro para metadados
            fwd_list = meta_entry.get('forward', [])
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fwd_list:
                loop_limit = min(total_frames, len(fwd_list)) - 1
            else:
                loop_limit = total_frames - 1

            ret, prev_frame = cap.read()
            if not ret: continue
            
            prev_tensor = self._process_frame(prev_frame)
            
            for i in range(loop_limit):
                ret, curr_frame = cap.read()
                if not ret: break
                
                curr_tensor = self._process_frame(curr_frame)
                
                # Pega ação do frame anterior
                action_id = self._default_mapper(meta_entry, i)
                
                # Yield: (Estado Atual, Ação Tomada, Próximo Estado)
                yield prev_tensor, action_id, curr_tensor
                
                prev_tensor = curr_tensor
                
            cap.release()

    def __iter__(self):
        return self._video_generator()

def create_dataloader(config_path, batch_size, max_videos=None):
    """Factory method"""
    dataset = MineRLDataset(config_path, max_videos=max_videos)
    # num_workers=0 é mais seguro para debugging, pode aumentar se tiver CPU sobrando
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)