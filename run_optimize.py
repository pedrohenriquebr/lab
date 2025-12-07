import subprocess
import argparse
import os
import time
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel optimization")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--restore", action="store_true", help="Continue from an existing study")
    args = parser.parse_args()
    
    # 1. Limpa o banco antigo (Começa estudo do zero)
    if args.restore:
        print("♻️ Continuando de um estudo existente...")
        
    if not args.restore and os.path.exists("optuna_db.sqlite3"):
        try:
            os.unlink("optuna_db.sqlite3")
            print("🗑️ Banco de dados antigo removido.")
        except PermissionError:
            print("⚠️ Não foi possível remover o banco (pode estar em uso). Continuando...")

    print(f"🚀 Iniciando {args.workers} trabalhadores em paralelo...")
    
    processes = []
    
    # 2. Inicia os processos
    for i in range(args.workers):
        print(f"   └─ Iniciando Worker {i+1}...")
        # Popen não bloqueia o script, ele lança e continua
        # Usamos sys.executable para garantir que usa o mesmo Python do venv
        p = subprocess.Popen([sys.executable, "optimize.py"])
        processes.append(p)
        
        # Pequeno delay para evitar conflito de criação do banco no milissegundo inicial
        time.sleep(2) 

    print("\n⏳ Todos os trabalhadores iniciados. Aguardando conclusão...")

    # 3. Espera todos terminarem
    for p in processes:
        p.wait()
        
    print("\n✅ Otimização Paralela Concluída!")