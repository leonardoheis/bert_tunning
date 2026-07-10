from src.schema import Hyperparams


def test_hyperparams_accepts_snake_case_construction() -> None:
    # Regression guard: Hyperparams(alias_generator=to_camel) without populate_by_name=True
    # rejects snake_case kwargs with a Pydantic ValidationError instead of accepting them --
    # exactly how src/training/pipeline.py constructs it (Hyperparams(model=..., batch_size=...)).
    # Broke every training run at the post-training reporting step until populate_by_name=True
    # was added; this test would have caught it.
    hyperparams = Hyperparams(
        model="beto",
        epochs=15,
        batch_size=8,
        grad_accum=8,
        effective_batch=64,
        learning_rate=2e-5,
        warmup_steps=100,
        precision="bf16",
        train_docs=1344,
        num_classes=9,
    )
    assert hyperparams.batch_size == 8  # noqa: PLR2004
    assert hyperparams.num_classes == 9  # noqa: PLR2004
