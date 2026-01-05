import time
import optuna
import mlflow
import numpy as np
from esp32robot.utils import get_git_info
from evaluate import run_evaluation
from train import train_agent

SESSION_ID = int(time.time())

def objective(trial: optuna.Trial):
    # ==========================================
    # 1. HIPERPARÂMETROS DA REDE (CÉREBRO)
    # ==========================================
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    gamma = trial.suggest_float("gamma", 0.90, 0.99) # Alto = Preocupado com o futuro
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.95, 0.995)
    
    # ==========================================
    # 2. PARÂMETROS DE SENSORES E FÍSICA (CORPO)
    # ==========================================
    # Latência: Tente manter baixa, mas deixe o Optuna explorar um pouco
    latency_steps = trial.suggest_int("latency_steps", 1, 3) 
    
    # Stack Size: Memória de curto prazo
    stack_size = 4
    
    # Threshold Frontal: O quão cedo ele considera "Bloqueado"?
    # 0.15 (Míope) até 0.45 (Visão de Águia)
    thresh_front = trial.suggest_float("threshold_blocked_front", 0.15, 0.45)
    
    # ==========================================
    # 3. TUNAGEM DE RECOMPENSAS (PSICOLOGIA)
    # ==========================================
    # Quanto dói bater? (-20 a -100)
    r_collision = trial.suggest_float("rew_collision", -100.0, -20.0, step=10.0)
    
    # Quanto ganha por andar? (0.5 a 3.0) - Menos ganância ajuda a não bater
    r_fwd = trial.suggest_float("rew_fwd", 0.5, 3.0, step=0.5)
    
    # Monta o dicionário de recompensas dinâmico
    dynamic_rewards = {
        'collision': r_collision,
        'penalties': {
            'indecision': -0.5,
            'idle': -5.0, # Mantemos punição alta pra ele não dormir
            'proximity_factor': 2.0
        },
        'states': {
            'in_corner': {'action_back': 2.0, 'default': -2.0},
            'blocked_front': {
                'action_back': 0.5,
                'action_turn': 2.0, # Girar é bom
                'action_fwd': -5.0, # Furar parede é ruim
                'default': -5.0
            },
            'clear_path': {
                'action_fwd': r_fwd,       # <--- Otimizado
                'action_stop': -10.0,
                'action_back': -5.0,
                'action_turn': -0.5
            }
        }
    }

    trial_name = f"opt_kamikaze_{trial.number:03d}"
    print(f"\n🔄 Trial {trial.number}: Front={thresh_front:.2f}, Coll={r_collision}, Fwd={r_fwd}")

    # ==========================================
    # 4. EXECUÇÃO
    # ==========================================
    with mlflow.start_run(run_name=trial_name, nested=True):
        mlflow.log_params(trial.params)
        
        # Treina
        agent, env, metrics = train_agent(
            model_name=trial_name,
            episodes=80,             # 80 eps para validar rápido
            max_steps=300,
            batch_size=batch_size,
            learning_rate=lr,      
            epsilon_decay=epsilon_decay,
            gamma=gamma,
            stack_size=stack_size,
            latency_steps=latency_steps,
            
            # Passando os parâmetros otimizados de ambiente
            threshold_blocked_front=thresh_front,
            threshold_corner=0.20, # Fixo
            threshold_proximity_danger=0.15, # Fixo
            threshold_idle_movement=0.02, # Fixo baixo
            rewards=dynamic_rewards, # <--- AQUI VAI A MÁGICA,
            
            world_model_path='mini_world_model.pth',
            world_model_n_categories=32,
            world_model_hidden_size= 64,
            
            headless=True
        )
        
        # Métricas de Treino
        avg_dist = np.mean(metrics['avg_distance_history'][-20:]) # Últimos 20 eps
        crash_rate = np.mean(metrics['crash_rate_hisptory'][-20:])
        
        # FAIL FAST: Se bater em mais de 30% dos episódios finais, penaliza muito
        if crash_rate > 0.30:
            print(f"💀 Alta taxa de colisão ({crash_rate:.2f}). Abortando trial.")
            return -500.0 - (crash_rate * 100)

        # 5. Avaliação (Prova Real)
        eval_reward, action_dict, success_rate = run_evaluation(
            model_name=trial_name, 
            latency_steps=latency_steps,
            stack_size=stack_size,
            episodes=10, 
            headless=True
        )
        
        # Score Final:
        # Queremos: Sucesso alto + Distância alta
        # O crash_rate já foi filtrado acima, mas penalizamos se houver algum crash na prova
        score = (success_rate * 1000.0) + (avg_dist * 10.0) + (eval_reward * 2.0)
        
        mlflow.log_metric("optuna_score", score)
        mlflow.log_metric("eval_success", success_rate)
        
        return score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    
    mlflow.set_experiment("ESP32_Optuna_Physics_Tuning")
    
    study = optuna.create_study(
        study_name="ESP32_Kamikaze_Fix",
        storage="sqlite:///optuna_db.sqlite3",
        direction="maximize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=args.trials)
    
    print("\n🏆 MELHORES PARÂMETROS PARA NÃO BATER:")
    print(study.best_params)