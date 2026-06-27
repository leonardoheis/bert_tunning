from src.training.models import ModelConfig

config = ModelConfig(
    name="xlm-roberta-base",
    hf_id="xlm-roberta-base",
    max_tokens=512,
    lr=2e-5,
    batch_size=8,
    grad_accum=8,
    force_fp32=False,
)
