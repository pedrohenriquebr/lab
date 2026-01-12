import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

SEQ_LEN = 6 
        
class MineRLDataset(IterableDataset):
    def __init__(self, root_path, img_size=64, action_mapper=None, max_videos=None, videos=None,skip_frames=10):
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), # Muda luz
        #     transforms.ToTensor(),
        # ])
        self.skip_frames = skip_frames
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
            
            raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Ajusta baseado no limite do metadata se existir
            meta_entry = self.metadata.get(vid_name, None)
            if meta_entry and 'forward' in meta_entry:
                raw_frames = min(raw_frames, len(meta_entry['forward']))
                
            
            step_size  = self.skip_frames + 1
            effective_frames = (raw_frames + step_size - 1) // step_size
            
            # Cada frame gera 1 amostra (exceto o último que não tem 'next')
            if effective_frames > SEQ_LEN:
                self.total_samples += (effective_frames - SEQ_LEN)
                
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
            
            if 'action_discrete' in meta_entry:
                return meta_entry['action_discrete'][frame_idx]

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
        """
        Generator modificado para Multistep Prediction COM FRAME SKIPPING.
        """

        
        for vid_name in self.videos:
            vid_path = os.path.join(self.root_path, vid_name)
            cap = cv2.VideoCapture(vid_path)
            
            meta_entry = self.metadata.get(vid_name, {})
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_buffer = []
            action_buffer = []
            
            frame_idx = 0
            while True:
                # --- FRAME SKIPPING (NOVO) ---
                # Lê e descarta frames para acelerar o tempo
                for _ in range(self.skip_frames):
                    success_skip, _ = cap.read()
                    if not success_skip:
                        break # Vídeo acabou durante o pulo
                    frame_idx += 1 # Importante: manter o índice de ação sincronizado!
                # -----------------------------

                ret, frame = cap.read()
                if not ret: break # Vídeo acabou
                
                # Processa frame
                tensor = self._process_frame(frame)
                frame_buffer.append(tensor)
                
                # Pega ação deste frame (agora sincronizado com o pulo)
                if frame_idx < total_frames - 1: 
                    action_id = self.action_mapper(meta_entry, frame_idx)
                    action_buffer.append(action_id)
                
                # Se encheu o buffer, solta a sequência
                if len(frame_buffer) == SEQ_LEN:
                    seq_frames = torch.stack(frame_buffer) 
                    
                    # Correção de segurança: as vezes o buffer de ação fica menor se o vídeo acabar
                    if len(action_buffer) >= SEQ_LEN - 1:
                        seq_actions = torch.tensor(action_buffer[:SEQ_LEN-1])
                        yield seq_frames, seq_actions
                    
                    frame_buffer.pop(0)
                    action_buffer.pop(0)
                
                frame_idx += 1
                
            cap.release()

    def __iter__(self):
        return self._video_generator()
    

def create_dataloader(config_path, videos, batch_size, img_size=64, max_videos=None, is_colab=False):
    """Factory method inteligente"""
    
    # 1. Configurações otimizadas para Colab/Linux vs Windows
    if is_colab or os.name == 'posix':
        num_workers = 4        # Colab aguenta bem 2 ou 4 workers
        prefetch_factor = 2    # Prepara batches na memória enquanto GPU treina
        pin_memory = True      # Acelera transferência para GPU
        persistent_workers = True # Mantém workers vivos (menos overhead)
    else:
        # Windows geralmente trava com workers > 0 em IterableDatasets simples
        num_workers = 0        
        prefetch_factor = None
        pin_memory = False
        persistent_workers = False

    print(f"⚙️ DataLoader Config: Workers={num_workers}, Pin={pin_memory}, ColabMode={is_colab}")

    dataset = MineRLDataset(config_path, videos=videos,max_videos=max_videos, img_size=img_size)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )