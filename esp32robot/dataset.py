from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from esp32robot.config import ACTION_MAP, CSV_FILE, DATASET_DIR, ESP_CMD_BASE, ESP_STREAM_URL, IMG_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, TIMEOUT

app_typer = typer.Typer()
import os
import csv
import time
import datetime
import threading
from flask import Flask, Response, request, jsonify
import requests

app = Flask(__name__)


# --- VARIÁVEIS GLOBAIS DE ESTADO ---
IS_RECORDING = False
CURRENT_FRAME = None  # Guarda a última imagem JPEG completa recebida
FRAME_LOCK = threading.Lock() # Para evitar conflito de escrita/leitura

# Garante que as pastas existem
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
    # Cria CSV com cabeçalho se não existir
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "image_path", "action_id", "action_name"])

print(f"🚀 Proxy de Gravação Iniciado!")
print(f"📂 Salvando dados em: {os.path.abspath(DATASET_DIR)}")

# --- FUNÇÕES AUXILIARES ---
def save_interaction(action_name):
    """Salva o frame atual e a ação no disco/csv"""
    global IS_RECORDING, CURRENT_FRAME
    
    if not IS_RECORDING:
        return

    with FRAME_LOCK:
        frame_data = CURRENT_FRAME

    if frame_data is None:
        print("⚠️ Tentativa de gravar, mas nenhum frame disponível ainda.")
        return

    # Gera nome único para o arquivo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"img_{timestamp}.jpg"
    filepath = os.path.join(IMG_DIR, filename)
    
    # 1. Salva Imagem JPG
    try:
        with open(filepath, 'wb') as f:
            f.write(frame_data)
    except Exception as e:
        print(f"❌ Erro ao salvar imagem: {e}")
        return

    # 2. Salva no CSV
    try:
        action_id = ACTION_MAP.get(action_name, -1)
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # Caminho relativo para facilitar treinamento em outros PCs
            rel_path = os.path.join("images", filename)
            writer.writerow([timestamp, rel_path, action_id, action_name])
            print(f"💾 [REC] Ação: {action_name} | Frame salvo: {filename}")
    except Exception as e:
        print(f"❌ Erro ao salvar CSV: {e}")

# --- ROTAS DE CONTROLE DE GRAVAÇÃO ---
@app.route('/record/start')
def start_recording():
    global IS_RECORDING
    IS_RECORDING = True
    return "🔴 Gravação INICIADA. Pilote o robô para gerar dados.", 200

@app.route('/record/stop')
def stop_recording():
    global IS_RECORDING
    IS_RECORDING = False
    return "⚪ Gravação PARADA.", 200

# --- ROTA 1: COMANDOS (Intercepta e Grava) ---
@app.route('/<action>')
def proxy_command(action):
    allowed = ['go', 'stop', 'back', 'left', 'right']
    
    if action not in allowed:
        return f"Ação '{action}' não reconhecida", 400

    # SE ESTIVER GRAVANDO, SALVA ANTES DE ENVIAR O COMANDO
    if IS_RECORDING and action != 'stop': 
        # Nota: Geralmente não gravamos 'stop' em behavioral cloning 
        # para evitar dataset desbalanceado (muitos stops), 
        # mas se quiser gravar tudo, remova o "and action != 'stop'"
        save_interaction(action)
    elif IS_RECORDING and action == 'stop':
        # Opcional: Gravar STOP também? Depende da sua estratégia.
        save_interaction(action)

    try:
        target_url = f"{ESP_CMD_BASE}/{action}"
        # Timeout curto para não travar a gravação
        resp = requests.get(target_url, timeout=TIMEOUT)
        return f"Comando {action} ok", resp.status_code
    except Exception as e:
        return "Erro ESP32", 502

# --- ROTA 2: VÍDEO (Sniffer de MJPEG) ---
@app.route('/stream')
def proxy_stream():
    try:
        req = requests.get(ESP_STREAM_URL, stream=True, timeout=5)
        
        def generate():
            global CURRENT_FRAME
            bytes_buffer = b''
            
            for chunk in req.iter_content(chunk_size=1024):
                bytes_buffer += chunk
                
                # Procura início (ffd8) e fim (ffd9) do JPEG
                a = bytes_buffer.find(b'\xff\xd8')
                b = bytes_buffer.find(b'\xff\xd9')
                
                if a != -1 and b != -1:
                    # Extrai o frame completo
                    jpg = bytes_buffer[a:b+2]
                    bytes_buffer = bytes_buffer[b+2:]
                    
                    # Atualiza a variável global para o gravador usar
                    with FRAME_LOCK:
                        CURRENT_FRAME = jpg
                        
                yield chunk

        return Response(generate(), 
                        mimetype='multipart/x-mixed-replace; boundary=123456789000000000000987654321')
    
    except Exception as e:
        print(f"❌ Erro no Stream: {e}")
        return "Stream Off", 502

@app_typer.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app_typer()
