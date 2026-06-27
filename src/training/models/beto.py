from src.training.models import ModelConfig

config = ModelConfig(
    name="beto",
    hf_id="dccuchile/bert-base-spanish-wwm-cased",
    max_tokens=512,
    lr=3e-5,
    batch_size=16,
    grad_accum=4,
    force_fp32=False,
)
