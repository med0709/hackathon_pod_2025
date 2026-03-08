from pathlib import Path

# Parâmetros do projeto
PROJECT_ROOT = Path(__file__).resolve().parent

# Data
PREDICTIONS_DIR = PROJECT_ROOT / 'data' / 'predictions' 

# Reports
METRICS_DIR = PROJECT_ROOT / 'reports' / 'metricas'
GRAPHICS_DIR = PROJECT_ROOT / 'reports' / 'graficos'

# models
MODELS_DIR     = PROJECT_ROOT / 'models'

print("Diretórios carregadas com sucesso")