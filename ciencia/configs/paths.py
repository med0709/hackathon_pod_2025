from pathlib import Path

# Parâmetros do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
PREDICTIONS_DIR = PROJECT_ROOT / 'data' / 'predictions' 
PROCESSED_DIR   = PROJECT_ROOT / 'data' / 'processed'
RAW_DIR         = PROJECT_ROOT / 'data' / 'raw'

# models
MODELS_DIR      = PROJECT_ROOT / 'models' / 'models' 
PARAMS_DIR      = PROJECT_ROOT / 'models' / 'best_params' 

# Artefatos
ARTIFACT_DIR    = PROJECT_ROOT / 'artifact'

# Reports
METRICS_DIR     = PROJECT_ROOT / 'reports' 

print("Diretórios carregados com sucesso")