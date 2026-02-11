from pathlib import Path

# Parâmetros do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
RAW_DIR         = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR   = PROJECT_ROOT / 'data' / 'processed'
PREDICTIONS_DIR = PROJECT_ROOT / 'data' / 'predictions' 

# Artefatos
ARTIFACT_DIR     = PROJECT_ROOT / 'artifact'

# Reports
METRICS_DIR = PROJECT_ROOT / 'reports'

# models
MODELS_DIR     = PROJECT_ROOT / 'models'

print("Diretórios carregadas com sucesso")