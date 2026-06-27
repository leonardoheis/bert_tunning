DOCS_ROOT = r"C:\Users\YourUser\Downloads\downloadsdocs\downloads"

# Model options (pick one):
#   "xlm-roberta-base"                  — recommended: stable multilingual, strong Spanish
#   "PlanTL-GOB-ES/roberta-base-bne"    — Spanish-only RoBERTa (National Library corpus)
#   "dccuchile/bert-base-spanish-wwm-cased"  — BETO: original Spanish BERT
#   "microsoft/deberta-v3-base"         — high accuracy but numerically unstable in practice
MODEL_NAME = "xlm-roberta-base"
MODEL_KEY = "xlm-roberta"   # key into src.training.models.MODEL_REGISTRY
OUTPUT_DIR = "./models/classiflow_model"
MAX_TOKENS = 512
CHUNK_STRATEGY = "first"  # "first" | "middle"
BATCH_SIZE = 8
GRAD_ACCUM = 8  # effective batch = 64
EPOCHS = 15
LR = 2e-5
FORCE_FP32 = False  # xlm-roberta trains fine in bf16/fp16
EARLY_STOP_PATIENCE = 5  # eval epochs without macro_f1 improvement before stopping

WANDB_ENTITY = "leonardo-a-heis"
WANDB_PROJECT = "bert_tunning"
SEED = 42
CACHE_PATH = "./data/classiflow_cache.parquet"

EXCLUDE_LABELS = {"convenios"}  # set()  # include everything

FOLDER_TO_LABEL = {
    "decretos": "decreto",
    "decreto_concejo_municipal": "decreto_concejo_municipal",
    "ordenanzas": "ordenanza",
    "decreto_ordenanzas": "decreto_ordenanza",
    "resoluciones": "resolucion",
    "resoluciones_concejo_municipal": "resolucion_concejo_municipal",
    "declaraciones_concejo_municipal": "declaracion_concejo_municipal",
    "convenios": "convenio",
}

API_HOST = "0.0.0.0"  # noqa: S104
API_PORT = 8000
