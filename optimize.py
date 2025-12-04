import optuna
import mlflow
from esp32robot.utils import get_git_info
from train import train_agent
import numpy as np 

def objective(trial: optuna.Trial):
    # 1. Sugestão de Parâmetros
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.95, 0.98)
    max_steps = trial.suggest_int('max_steps', 200, 300, step=25)
    
    # 2. Nome do Modelo (Padrão ID Curto)
    # Ex: opt_trial_005
    trial_name = f"opt_reward_tuning_trial_{trial.number:03d}"
    
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
            episodes=30,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=lr,      
            epsilon_decay=epsilon_decay 
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
        else:
            score = avg_reward + (avg_distance*10.0)
            mlflow.log_metric("fail_reason", 0)
        
        # Loga a métrica alvo no MLflow
        mlflow.log_metrics({
            "optuna_score": score,
            "final_dist": avg_distance,
            "final_idle": avg_idle_rate
        })

    return score

if __name__ == "__main__":
    # Configura o Experimento no MLflow
    mlflow.set_experiment("ESP32_Optuna_Tuning")
    
    # Inicia a Run "Pai"
    with mlflow.start_run(run_name="Optuna_Session_v3_Reward_Tuning"):
        git_commit, git_branch = get_git_info()
        mlflow.set_tag("git.commit", git_commit)
        mlflow.set_tag("git.branch", git_branch)
        mlflow.set_tag("user", "Pedro")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        
        # Loga os melhores parâmetros na Run Pai
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_reward", study.best_value)
        
        print("\n🏆 MELHORES PARÂMETROS ENCONTRADOS:")
        print(study.best_params)