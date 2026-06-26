DOCS_ROOT = r"C:\Users\YourUser\Downloads\downloadsdocs\downloads"
MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = "./models/classiflow_deberta_model"
MAX_TOKENS = 512
CHUNK_STRATEGY = "first"  # "first" | "middle"
BATCH_SIZE = 4  # 8
GRAD_ACCUM = 16  # 8  # effective batch = 8 × 8 = 64
EPOCHS = 15
LR = 1e-5
FORCE_FP32 = True  # DeBERTa-v3 disentangled attention overflows in bf16/fp16

WANDB_ENTITY = "leonardo-a-heis"
WANDB_PROJECT = "bert_tunning"
SEED = 42
CACHE_PATH = "./classiflow_cache.parquet"

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
