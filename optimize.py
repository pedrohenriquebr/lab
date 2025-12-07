import time
import optuna
import mlflow
from esp32robot.utils import get_git_info
from evaluate import run_evaluation
from train import train_agent
import numpy as np

SESSION_ID = int(time.time())

def objective(trial: optuna.Trial):
    # 1. Sugestão de Parâmetros (Ajustados para Stacking + Latência)
    
    # Learning Rate (DQN)
    lr = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True)
    
    # Batch Size
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    
    # Fator de Desconto (Gamma) - Importante para latência
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    
    # Epsilon Decay
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.90, 0.99)
    
    # Duração do episódio
    max_steps = trial.suggest_int('max_steps', 250, 400, step=50)
    
    stack_size = trial.suggest_int("stack_size", 3, 8)
    latency_steps = 5
    
    # Rotation Penalty (Fixamos em um valor baixo apenas para evitar dança)
    # Não otimizamos mais isso, pois o anti-jitter do env já cuida do grosso
    ROTATION_PENALTY = 0.01 
    
    # Nome do Trial
    trial_name = f"opt_v5_latency_{SESSION_ID}_trial_{trial.number:03d}"
    
    print(f"\n🔄 Iniciando Trial {trial.number}: LR={lr:.5f}, Gamma={gamma:.3f}, Batch={batch_size}")

    # 3. Integração MLflow (Nested Run)
    # nested=True faz essa run ficar "dentro" da run principal do Optuna
    with mlflow.start_run(run_name=trial_name, nested=True):
        
        # Loga os parâmetros que o Optuna escolheu
        mlflow.log_params(trial.params)
        
        # Roda o treino passando os parâmetros dinâmicos
        # Nota: Seu train_agent precisa aceitar esses argumentos novos!
        agent, env, metrics = train_agent(
            model_name=trial_name,  # O nome limpo
            episodes=60,            # 60 episódios para dar tempo de aprender latência
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=lr,      
            epsilon_decay=epsilon_decay,
            rotation_penalty=ROTATION_PENALTY,
            headless=True,
            gamma=gamma,
            stack_size=stack_size,
            latency_steps=latency_steps
            # O train_agent precisa passar gamma para o agente internamente
            # Se não passar, vamos confiar no padrão ou injetar (ver abaixo)
        )
        
        # Métricas do Treino
        avg_reward = np.mean(metrics['reward_history']) 
        avg_crash_rate = np.mean(metrics['crash_rate_history']) 
        avg_idle_rate = np.mean(metrics['idle_rate_history']) 
        avg_distance = np.mean(metrics['avg_distance_history'])
        
        # 4. Filtro de Qualidade (Fail Fast)
        # Se o histórico for preguiçoso, retorna um valor muito baixo e aborta
        if avg_distance < 1.0 or avg_idle_rate > 0.5:
            avg_reward = -1000.0
            mlflow.log_metric("fail_reason", 1)
            return avg_reward
        else:
            mlflow.log_metric("fail_reason", 0)
        
        
        # 5. Avaliação (Prova Real)
        eval_reward, action_dict, succes_rate = run_evaluation(
            model_name=trial_name, 
            latency_steps=latency_steps,
            stack_size=stack_size,
            episodes=10, # 10 episódios de teste
            headless=True, # Sem janela para ser rápido,
            delay=0.00
        )
        
        # 3. Verifica Viés (Bias Check)
        total_actions = sum(action_dict.values())
        if total_actions > 0:
            pct_right = action_dict[4] / total_actions
            pct_left = action_dict[3] / total_actions
        else:
            pct_right = 0.0
            pct_left = 0.0
        
        mlflow.log_metrics({
            "test_act_stop_pct": action_dict[0] / total_actions if total_actions > 0 else 0.0,
            "test_act_fwd_pct": action_dict[1] / total_actions if total_actions > 0 else 0.0,
            "test_act_back_pct": action_dict[2] / total_actions if total_actions > 0 else 0.0,
            "test_act_left_pct": action_dict[3] / total_actions if total_actions > 0 else 0.0,
            "test_act_right_pct": action_dict[4] / total_actions if total_actions > 0 else 0.0,
            "test_success_rate": succes_rate,
            "test_avg_reward": eval_reward
        })
        
        # Penalidade de Viés Extremo
        if pct_right > 0.8 or pct_left > 0.8: # Um pouco mais tolerante (0.8)
            mlflow.log_metric("bias_penalty", 1)
            return -500.0
        
        # Supondo Reward máx ~100 para normalizar
        norm_reward = eval_reward / 100.0 
        
        action_counts = list(action_dict.values())
        if total_actions > 0:
            action_std = np.std(action_counts) / total_actions
        else:
            action_std = 1.0 
        
        # --- A Fórmula Mágica Simplificada ---
        # Prioridade total: Chegar no alvo (succes_rate)
        # Desempate: Distância percorrida (avg_distance)
        score = (
            (1.0 * norm_reward) +       # Performance Geral
            (50.0 * succes_rate) +      # Ojetivo Final (Peso alto!)
            (5.0 * avg_distance) -      # Bônus por andar longe
            (2.0 * avg_idle_rate)       # Penalidade leve por parar
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
    mlflow.set_experiment("ESP32_Latency_Stacking_Tuning")
    
    # Inicia a Run "Pai"
    with mlflow.start_run(run_name=f"Optuna_Session_v6_Latency_{SESSION_ID}"):
        git_commit, git_branch = get_git_info()
        mlflow.set_tag("git.commit", git_commit)
        mlflow.set_tag("git.branch", git_branch)
        mlflow.set_tag("user", "Pedro")
        
        storage_url = "sqlite:///optuna_db.sqlite3"
        
        study = optuna.create_study(
            study_name="ESP32_Latency_Stacking",
            storage=storage_url,
            direction="maximize",
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=2)
        
        # Loga os melhores parâmetros na Run Pai
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_reward", study.best_value)
        
        print("\n🏆 MELHORES PARÂMETROS ENCONTRADOS:")
        print(study.best_params)