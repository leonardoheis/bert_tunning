from .config import ModelConfig

config = ModelConfig(
    name="multilingual-minilm",
    hf_id="microsoft/Multilingual-MiniLM-L12-H384",
    max_tokens=512,
    lr=3e-5,
    batch_size=32,
    grad_accum=2,
    force_fp32=False,
)
