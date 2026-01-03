from pathlib import Path

import cv2
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
import typer

from esp32robot.config import PROCESSED_DATA_DIR

app = typer.Typer()


class DepthEstimator:
    def __init__(self, use_gpu=False):
        self.device = torch.device("cpu")
        print(f"Carregando MiDaS Small (Rápido para CPU)...")

        # Carrega o MiDaS Small via Torch Hub (Leve e Rápido)
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)
        self.model.eval()

        # Transformações necessárias
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def predict(self, img_rgb):
        # Aplica transformação
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            # Redimensiona para o tamanho original DA IMAGEM DE ENTRADA
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2], # Garante que saia do mesmo tamanho que entrou
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normaliza 0-255
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6) # +1e-6 evita div por zero

        return (depth_norm * 255).astype(np.uint8)

class VisionProcessor:
    def __init__(self, width=150, height=150):
        self.width = width
        self.height = height
        self.depth_estimator = DepthEstimator()

    def filter_target(self, img):
        # Filtro de cor vermelha (Alvo)
        b, g, r = cv2.split(img)
        return ((r.astype(int) > g.astype(int) + 40) & (r.astype(int) > b.astype(int) + 40)).astype(np.uint8) * 255

    def filter_obstacle(self, img):
        # Filtro básico de obstáculos (não usado diretamente no state 2D, mas útil pra debug)
        depth = self.depth_estimator.predict(img)
        return (depth > 150).astype(np.uint8) * 255

    def compute_state_2d(self, depth_map):
        """
        Converte o mapa de profundidade em 3 sensores virtuais.
        Ajustado para ignorar o chão e ruídos de iluminação.
        """
        h, w = depth_map.shape
        
        # --- AJUSTE 1: CORTAR O CHÃO (CROP) ---
        # O robô não precisa ver o chão imediato, ele precisa ver paredes à frente.
        roi_h = int(h * 0.7) 
        depth_roi = depth_map[:roi_h, :] # Pega só do topo até 70% da altura
        
        # Divide a imagem cortada em 3 colunas
        cols = np.hsplit(depth_roi, 3)
        
        sensors = []
        for col in cols:
            # --- AJUSTE 2: USAR PERCENTIL EM VEZ DE MAX ---
            # np.max pega 1 pixel de ruído e estraga tudo.
            # np.percentile(col, 95) pega os 5% mais brilhantes (mais robusto)
            # ou np.mean(col) se quiser uma média geral.
            intensity = np.percentile(col, 95) 
            
            # --- AJUSTE 3: CALIBRAÇÃO DE LIMIAR (OFFSET) ---
            # Se a intensidade for menor que um piso (ex: 50), considera longe.
            # Isso ajuda a eliminar o "ruído de fundo" da iluminação.
            
            # Limiar de corte: Abaixo de 60 de brilho, considere 0 (Escuro/Longe)
            if intensity < 60:
                intensity = 0
            
            # Normaliza (0 a 1)
            # Ajustamos o denominador: em vez de 255, usamos 200 como "perto demais"
            # para dar mais sensibilidade.
            MAX_VAL = 160.0
            intensity_norm = min(1.0, intensity / MAX_VAL)
            
            # Inverte: 1.0 (Livre) -> 0.0 (Colado)
            # Adicionamos um fator de correção exponencial para não cair pra zero tão rápido
            distance = 1.0 - intensity_norm
            
            sensors.append(distance)
            
        return tuple(sensors)
            
    def compute_state(self, mask_target, mask_obstacle, depth_map):
        """
        NOVA LÓGICA: Detecta alvo mesmo com poucos pixels
        E distâncias são SEMPRE calculadas (não mascaradas)
        """
        rows_target = np.vsplit(mask_target, 3)
        rows_obstacle = np.vsplit(mask_obstacle, 3)
        dists = self.calculate_grid_distances(depth_map)

        state, idx = [], 0
        for r in range(3):  # Para cada linha
            cols_target = np.hsplit(rows_target[r], 3)
            cols_obstacle = np.hsplit(rows_obstacle[r], 3)

            for c in range(3):  # Para cada coluna
                # 1. Análise do TARGET (vermelho) - AGORA COM 3 NÍVEIS
                target_pixels = np.count_nonzero(cols_target[c])
                total_pixels = cols_target[c].size
                target_ratio = target_pixels / total_pixels

                # 2. Análise do OBSTÁCULO (perto)
                obstacle_pixels = np.count_nonzero(cols_obstacle[c])
                obstacle_ratio = obstacle_pixels / total_pixels

                # 3. Decisão baseada em prioridade E intensidade
                val = 0

                # Se tem ALVO VISÍVEL (mesmo pouco), prioridade máxima
                if target_ratio > 0.05:  # 🔴 REDUZIDO de 20% para 1%!
                    val = 4  # Alvo detectado

                # Se tem OBSTÁCULO PRÓXIMO, prioridade alta
                elif obstacle_ratio > 0.05:  # 5% de pixels brancos no depth
                    val = 3  # Perigo iminente

                # Se não tem obstáculo, use distância
                else:
                    d = dists[idx]
                    if d > 10.0: val = 0      # Livre
                    elif d > 5.0: val = 1     # Atenção
                    elif d > 1.5: val = 2     # Perigo
                    else: val = 3             # Colisão iminente

                state.append(val)
                idx += 1

        return tuple(state)

    def calculate_grid_distances(self, depth_map):
        rows = np.vsplit(depth_map, 3)
        distances = []
        for r in range(3):
            cols = np.hsplit(rows[r], 3)
            for c in range(3):
                avg_intensity = np.mean(cols[c])
                dist = 10 if avg_intensity < 1 else 200.0 / (avg_intensity + 0.1)
                distances.append(dist)
        return distances



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
