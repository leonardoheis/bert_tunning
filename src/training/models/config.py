from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    hf_id: str
    max_tokens: int
    lr: float
    batch_size: int
    grad_accum: int
    force_fp32: bool
