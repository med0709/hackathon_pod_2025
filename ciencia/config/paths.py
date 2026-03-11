from pathlib import Path

# Parâmetros do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
RAW_DIR         = PROJECT_ROOT / '01_data' / 'raw'
PROCESSED_DIR   = PROJECT_ROOT / '01_data' / 'processed'
PREDICTIONS_DIR = PROJECT_ROOT / '01_data' / 'predictions' 

# Artefatos
ARTIFACT_DIR     = PROJECT_ROOT / '04_artifact'

# Reports
METRICS_DIR = PROJECT_ROOT / '06_reports'

# models
MODELS_DIR     = PROJECT_ROOT / '05_models'

print("Diretórios carregadas com sucesso")