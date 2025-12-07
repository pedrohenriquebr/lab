import time
import optuna
import mlflow
import numpy as np
import os
from finetune import run_finetuning
from evaluate import run_evaluation
from esp32robot.config import MODELS_DIR
import argparse

SESSION_ID = int(time.time())

# MODELO BASE QUE VOCÊ QUER MELHORAR
BASE_MODEL_NAME = "opt_v5_latency_1765128725_trial_010"

# Parâmetros estruturais do modelo base (NÃO MUDE ISSO, tem que ser igual ao treino original)
STACK_SIZE = 4
LATENCY_STEPS = 5

def objective(trial: optuna.Trial):
    # --- PARÂMETROS PARA OTIMIZAR ---
    
    # 1. Learning Rate (Delicado vs Agressivo)
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.90, 0.99)
    
    epsilon_start = trial.suggest_float("epsilon_start", 0.1, 0.4)
    
    # 2. Quanto tempo treinar (Mais tempo = Mais refino?)
    episodes = trial.suggest_int("episodes", 50, 150, step=25)
    
    
    # Nome do novo modelo candidato
    new_model_name = f"ft_v5.1_{SESSION_ID}_trial_{trial.number:03d}"
    
    print(f"\n🔧 Iniciando Fine-Tune Trial {trial.number}: LR={lr:.2e}")

    # --- EXECUÇÃO ---
    with mlflow.start_run(run_name=new_model_name, nested=True):
        mlflow.log_params(trial.params)
        
        # 1. RODA O FINE-TUNING
        # Precisamos adaptar o run_finetuning para aceitar o jitter_penalty customizado
        # (Veja nota abaixo sobre adaptação no finetune.py se necessário, 
        # mas por hora vamos assumir que o run_finetuning usa o padrão ou você ajustou)
        
        # Para injetar o jitter_penalty sem mudar a assinatura do run_finetuning, 
        # você pode passar via 'correction_weights' ou similar, mas o ideal é 
        # atualizar o finetune.py para aceitar argumentos extras de ambiente.
        
        # Aqui, vamos usar a função run_finetuning padrão, focando em LR e Eps.
        # Se quiser otimizar o jitter, precisa alterar o finetune.py para receber esse argumento.
        
        run_finetuning(
            base_model_name=BASE_MODEL_NAME,
            new_model_name=new_model_name,
            episodes=episodes,
            stack_size=STACK_SIZE,
            latency_steps=LATENCY_STEPS,
            learning_rate = lr,
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay
        )
        
        # 2. AVALIAÇÃO (O Teste de Fogo)
        eval_reward, action_dict, success_rate = run_evaluation(
            model_name=f"finetuned/{new_model_name}", # Ele salva na pasta finetuned/
            episodes=50,
            headless=True,
            stack_size=STACK_SIZE,
            latency_steps=LATENCY_STEPS
        )
        
        # Métricas
        total_actions = sum(action_dict.values())
        if total_actions > 0:
            pct_left = action_dict[3] / total_actions
            pct_right = action_dict[4] / total_actions
        else:
            pct_left = 0
            pct_right = 0
            
      
        
        # Score Final: Prioridade total para SUCESSO
        score = (success_rate * 60 + eval_reward*40)/100
        
        mlflow.log_metrics({
            "eval_success": success_rate,
            "eval_reward": eval_reward,
            "bias_left": pct_left,
            "bias_right": pct_right,
            "final_score": score
        })
        return score

if __name__ == "__main__":
    mlflow.set_experiment("ESP32_Fine_Tuning_Optuna")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, help="Número de trials para otimização", required=True)
    args = parser.parse_args()
    
    # Nome da Run Pai (Fixa ou dinâmica se quiser várias sessões)
    parent_run_name = f"FT_Session_v5_{SESSION_ID}"
    
    # Inicia a Run Pai (Só para logar o melhor no final)
    # Em modo paralelo, cada processo tentará criar/entrar.
    # O ideal em paralelo é NÃO ter run pai global envolvendo o optimize,
    # mas sim cada trial ser uma run. O Optuna já gerencia isso.
    
    # URL do Banco Compartilhado
    storage_url = "sqlite:///optuna_ft.sqlite3"
    study_name = "ESP32_FT_Study"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True # Permite múltiplos processos
    )
    
    study.optimize(objective, n_trials=args.trials)
    
    print("\n🏆 MELHOR ATÉ AGORA (Neste Worker):")
    print(study.best_params)
    print(f"Melhor Modelo Salvo: ft_v5.1_{SESSION_ID}_trial_{study.best_trial.number:03d}.pth")