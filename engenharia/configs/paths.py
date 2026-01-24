from pathlib import Path

# Parâmetros do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
PREDICTIONS_DIR = PROJECT_ROOT / 'data' / 'predictions' 
PROCESSED_DIR   = PROJECT_ROOT / 'data' / 'processed'
RAW_DIR         = PROJECT_ROOT / 'data' / 'raw'

# Reports
METRICS_DIR = PROJECT_ROOT / 'reports' / 'metrics'

# models
MODELS_DIR     = PROJECT_ROOT / 'models'

# Artefatos
ARTIFACT_DIR     = PROJECT_ROOT / 'artifact'

print("Diretórios carregados com sucesso")
