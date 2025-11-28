from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
from esp32robot.agents import QAgent2D
from esp32robot.config import FIGURES_DIR, PROCESSED_DATA_DIR
import numpy as np
app = typer.Typer()


def plot_training_results(agent: QAgent2D):
    if not agent.use_dqn or not agent.loss_history:
        return
    
    # backend 'Agg' é essencial para não tentar abrir janelas GUI e travar o PyGame
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt

    # Cria figura nova a cada vez para evitar memory leak do Matplotlib
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    # ==========================================
    # GRÁFICO 1: LOSS
    # ==========================================
    ax[0].plot(agent.loss_history, label='Loss (MSE)', color='red', alpha=0.4)

    if len(agent.loss_history) > 50:
        window = 50
        moving_avg_loss = np.convolve(agent.loss_history, np.ones(window)/window, mode='valid')
        ax[0].plot(range(window-1, len(agent.loss_history)), moving_avg_loss, label='Média Móvel', color='darkred', linewidth=2)

    ax[0].set_title('Evolução do Erro (Loss)')
    ax[0].set_ylabel('Erro (MSE)')
    ax[0].legend()
    ax[0].grid(True)

    # ==========================================
    # GRÁFICO 2: RECOMPENSAS
    # ==========================================
    if hasattr(agent, 'reward_history') and agent.reward_history:
        ax[1].plot(agent.reward_history, label='Reward (Passo a Passo)', color='green', alpha=0.3)

        if len(agent.reward_history) > 50:
            window = 50
            moving_avg_rew = np.convolve(agent.reward_history, np.ones(window)/window, mode='valid')
            ax[1].plot(range(window-1, len(agent.reward_history)), moving_avg_rew, label='Tendência (Média)', color='blue', linewidth=2)

        ax[1].set_title('Histórico de Recompensas')
        ax[1].set_xlabel('Passos de Treinamento (Steps)')
        ax[1].set_ylabel('Recompensa')
        ax[1].legend()
        ax[1].grid(True)
    else:
        ax[1].text(0.5, 0.5, "Sem dados de reward_history", ha='center')

    fig.tight_layout()
    
    # SALVA E FECHA (Não usa plt.show)
    try:
        plt.savefig("training_charts.png")
    except:
        pass
    finally:
        plt.close(fig) # CRÍTICO: Libera a memória e a thread para o PyGame
        plt.close('all') 
    
@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
