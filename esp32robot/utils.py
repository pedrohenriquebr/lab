import subprocess

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