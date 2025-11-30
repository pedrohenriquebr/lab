import optuna
import mlflow
from esp32robot.utils import get_git_info
from train import train_agent

def objective(trial: optuna.Trial):
    # 1. Sugestão de Parâmetros
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.90, 0.999)
    max_steps = trial.suggest_int('max_steps', 50, 300, step=50) # Aumentei o min para dar tempo de aprender
    
    # 2. Nome do Modelo (Padrão ID Curto)
    # Ex: opt_trial_005
    trial_name = f"opt_trial_{trial.number:03d}"
    
    print(f"\n🔄 Iniciando Trial {trial.number}: LR={lr:.5f}, Batch={batch_size}")

    # 3. Integração MLflow (Nested Run)
    # nested=True faz essa run ficar "dentro" da run principal do Optuna
    with mlflow.start_run(run_name=trial_name, nested=True):
        
        # Loga os parâmetros que o Optuna escolheu
        mlflow.log_params(trial.params)
        
        # Roda o treino passando os parâmetros dinâmicos
        # Nota: Seu train_agent precisa aceitar esses argumentos novos!
        agent, env, history = train_agent(
            model_name=trial_name,  # O nome limpo
            episodes=35,            # Treino curto para ser rápido
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=lr,      
            epsilon_decay=epsilon_decay 
        )
        
        # 4. Define a Métrica de Sucesso (Média dos últimos rewards)
        # Se o histórico for vazio (erro), retorna um valor muito baixo
        if not history:
            avg_reward = -9999
        else:
            avg_reward = sum(history[-10:]) / 10 if len(history) >= 10 else sum(history) / len(history)
        
        # Loga a métrica alvo no MLflow
        mlflow.log_metric("avg_reward_final", avg_reward)
        
        # (Opcional) Loga o modelo no MLflow se quiser guardar todos (ocupa espaço!)
        # mlflow.log_artifact(f"models/{trial_name}.pth")

    return avg_reward

if __name__ == "__main__":
    # Configura o Experimento no MLflow
    mlflow.set_experiment("ESP32_Optuna_Tuning")
    
    # Inicia a Run "Pai"
    with mlflow.start_run(run_name="Optuna_Session_v2"):
        git_commit, git_branch = get_git_info()
        mlflow.set_tag("git.commit", git_commit)
        mlflow.set_tag("git.branch", git_branch)
        mlflow.set_tag("user", "Pedro")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10) # Roda 20 testes
        
        # Loga os melhores parâmetros na Run Pai
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_reward", study.best_value)
        
        print("\n🏆 MELHORES PARÂMETROS ENCONTRADOS:")
        print(study.best_params)