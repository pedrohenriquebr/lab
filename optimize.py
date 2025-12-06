import time
import optuna
import mlflow
from esp32robot.utils import get_git_info
from evaluate import run_evaluation
from train import train_agent
import numpy as np

SESSION_ID = int(time.time())

def objective(trial: optuna.Trial):
    # 1. Sugestão de Parâmetros (LR Aumentado!)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True) 
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    episodes = trial.suggest_int("episodes", 50, 150, step=25) # Dei um pouco mais de tempo
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.90, 0.98, step=0.01)
    max_steps = trial.suggest_int('max_steps', 200, 400, step=50)
    # rot_penalty = trial.suggest_float("rotation_penalty", 0.0005, 0.02, step=0.0005)
    
    # 2. Nome do Modelo (Padrão ID Curto)
    # Ex: opt_trial_005
    trial_name = f"opt_v4_{SESSION_ID}_trial_{trial.number:03d}"
    
    print(f"\n🔄 Iniciando Trial {trial.number}: LR={lr:.5f}, Batch={batch_size}")

    # 3. Integração MLflow (Nested Run)
    # nested=True faz essa run ficar "dentro" da run principal do Optuna
    with mlflow.start_run(run_name=trial_name, nested=True):
        
        # Loga os parâmetros que o Optuna escolheu
        mlflow.log_params(trial.params)
        
        # Roda o treino passando os parâmetros dinâmicos
        # Nota: Seu train_agent precisa aceitar esses argumentos novos!
        agent, env, metrics = train_agent(
            model_name=trial_name,  # O nome limpo
            episodes=episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=lr,      
            epsilon_decay=epsilon_decay,
            rotation_penalty=0,
            headless=True
        )
        
        avg_reward = np.mean(metrics['reward_history']) 
        avg_crash_rate = np.mean(metrics['crash_rate_history']) 
        avg_idle_rate = np.mean(metrics['idle_rate_history']) 
        avg_distance = np.mean(metrics['avg_distance_history'])
        
        # 4. Define a Métrica de Sucesso (Média dos últimos rewards)
        # Se o histórico for vazio (erro), retorna um valor muito baixo
        
        score = 0
        if avg_distance < 1.0 or avg_idle_rate > 0.5:
            avg_reward = -1000.0
            mlflow.log_metric("fail_reason", 1)
            return avg_reward
        else:
            mlflow.log_metric("fail_reason", 0)
        
        
        
        
        eval_reward, action_dict, succes_rate = run_evaluation(
            model_name=trial_name, 
            episodes=10, # 10 episódios de teste
            headless=True, # Sem janela para ser rápido,
            delay=0.00
        )
        
        
        
        
        # 3. Verifica Viés (Bias Check)
        total_actions:int = sum(action_dict.values())
        pct_right: float = 0.0
        
        mlflow.log_metrics({
            "test_act_stop_pct": action_dict[0] / total_actions if total_actions > 0 else 0.0,
            "test_act_fwd_pct": action_dict[1] / total_actions if total_actions > 0 else 0.0,
            "test_act_back_pct": action_dict[2] / total_actions if total_actions > 0 else 0.0,
            "test_act_left_pct": action_dict[3] / total_actions if total_actions > 0 else 0.0,
            "test_act_right_pct": action_dict[4] / total_actions if total_actions > 0 else 0.0,
            "test_success_rate": succes_rate,
            "test_avg_reward": eval_reward
        })
        
        
        if total_actions > 0:
            pct_right = action_dict[4] / total_actions
        else:
            pct_right = 0.0
        
        if pct_right > 0.5:
            mlflow.log_metric("bias_penalty", 1)
            return -500.0
        
        # Supondo Reward máx ~100
        norm_reward = eval_reward / 100.0 
        
        action_counts = list(action_dict.values())
        total_actions_eval = sum(action_dict.values())
        action_std = 0.0
        if total_actions_eval > 0:
            action_counts = list(action_dict.values())
            action_std = np.std(action_counts) / total_actions_eval
        else:
            action_std = 1.0 # Punição máxima se não fez nada
        
        # A Fórmula Mágica Simplificada
        score = (
            (1.0 * norm_reward) +       # Performance Geral
            (2.0 * succes_rate) -       # Ojetivo Final (Peso alto!)
            (1.0 * avg_idle_rate)
        )
        
        
        # Loga a métrica alvo no MLflow
        mlflow.log_metrics({
            "optuna_score": score,
            "final_dist": avg_distance,
            "final_idle": avg_idle_rate,
            "norm_reward": norm_reward,
            "action_std": action_std
        })
    
       
        
    return score

if __name__ == "__main__":
    # Configura o Experimento no MLflow
    mlflow.set_experiment("ESP32_Optuna_Tuning")
    
    # Inicia a Run "Pai"
    with mlflow.start_run(run_name="Optuna_Session_v4.1_Reward_Tuning"):
        git_commit, git_branch = get_git_info()
        mlflow.set_tag("git.commit", git_commit)
        mlflow.set_tag("git.branch", git_branch)
        mlflow.set_tag("user", "Pedro")
        storage_url = "sqlite:///optuna_db.sqlite3"
        
        study = optuna.create_study(
            study_name="ESP32_Optuna_Tuning",
            
            storage=storage_url,
            direction="maximize",
            load_if_exists=True)
        
        
        study.optimize(objective, n_trials=15)
        
        # Loga os melhores parâmetros na Run Pai
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_reward", study.best_value)
        
        print("\n🏆 MELHORES PARÂMETROS ENCONTRADOS:")
        print(study.best_params)