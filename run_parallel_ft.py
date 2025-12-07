import subprocess
import argparse
import os
import sys
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Número de processos paralelos")
    parser.add_argument("--restore", action="store_true", help="Continuar de um estudo existente")
    parser.add_argument("--trials", type=int, default=20, help="Número de trials por worker")
    args = parser.parse_args()
    
    
    if args.restore:
        print("♻️ Continuando de um estudo existente...")
    # Limpa banco antigo se quiser começar do zero
    if not args.restore and os.path.exists("optuna_ft.sqlite3"):
        print("🧹 Removendo banco antigo optuna_ft.sqlite3...")
        # try: os.unlink("optuna_ft.sqlite3")
        # except: pass

    print(f"🚀 Iniciando {args.workers} workers de Fine-Tuning...")
    
    procs = []
    for i in range(args.workers):
        # Chama o script de otimização
        cmd = [sys.executable, "optimize_finetune.py", "--trials", str(args.trials)]
        p = subprocess.Popen(cmd)
        procs.append(p)
        time.sleep(2) # Delay para evitar conflito na criação do DB

    for p in procs:
        p.wait()
        
    print("✅ Todos os workers finalizaram.")