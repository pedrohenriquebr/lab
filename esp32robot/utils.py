import subprocess

import torch
import torch.nn as nn

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode('utf-8')
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('utf-8')
        return commit, branch
    except:
        return "unknown", "unknown"
    
    

import yaml
import os

def load_config(config_path="config.yaml"):
    """Carrega o arquivo YAML e retorna um dicionário"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    print(f"📄 Configuração carregada de: {config_path}")
    return config


class ConfigurationParameters:
    def __init__(self, environment_config: dict, agent_config: dict, world_model_config: dict, training_config: dict):
        self.environment_config = environment_config
        self.agent_config = agent_config
        self.world_model_config = world_model_config
        self.training_config = training_config
        
        


class DynamicLossTuner(nn.Module):
    def __init__(self, n_losses=4):
        """
        Ajusta automaticamente os pesos de 4 losses:
        0: Reconstruction
        1: Prediction
        2: Inverse Dynamics
        3: Entropy/KL
        """
        super().__init__()
        # Inicializa os parâmetros de variância (log variance) com 0
        # Eles são treináveis pelo otimizador!
        self.log_vars = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses_dict):
        """
        Entrada: Dicionário ou Lista de Losses (escalares)
        Retorna: Loss Total ponderada automaticamente
        """
        # Organiza as losses na ordem: [Rec, Pred, Inv, Ent]
        # Garante que estão no mesmo device
        losses = torch.stack([
            losses_dict['rec'],
            losses_dict['pred'],
            losses_dict['inv'],
            losses_dict['ent']
        ]).to(self.log_vars.device)

        # Fórmula Mágica de Kendall (Multi-Task Learning using Uncertainty)
        # Loss = (1 / 2*var) * Loss_Original + log(std)
        # Se a var aumenta (peso diminui), o termo log(std) pune o modelo.
        # Isso cria um equilíbrio perfeito matemático.
        
        precision = torch.exp(-self.log_vars)
        weighted_losses = precision * losses + self.log_vars
        
        return weighted_losses.sum(), precision