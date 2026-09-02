# Model Provenance

The three models registered in `src/training/models/`. All are pulled automatically from
the Hugging Face Hub by `transformers` via `from_pretrained(hf_id)` — there is no manual
download step.

| Registry key (CLI `--model`) | `ModelConfig.name` | HF ID (`hf_id`) | Download | Origin paper |
|---|---|---|---|---|
| `xlm-roberta` | `xlm-roberta-base` | `xlm-roberta-base` | https://huggingface.co/xlm-roberta-base | Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* Facebook AI — https://arxiv.org/abs/1911.02116 |
| `beto` | `beto` | `dccuchile/bert-base-spanish-wwm-cased` | https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased | Cañete, J. et al. (2020). *Spanish Pre-Trained BERT Model and Evaluation Data.* PML4DC @ ICLR, Universidad de Chile — https://users.dcc.uchile.cl/~jperez/papers/pml4dc2020.pdf |
| `minilm` | `multilingual-minilm` | `microsoft/Multilingual-MiniLM-L12-H384` | https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384 | Wang, W. et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.* NeurIPS, Microsoft — https://arxiv.org/abs/2002.10957 |

## Notes

- Registry lives in `src/training/models/__init__.py`; each model is one file exporting a
  `config: ModelConfig`.
- `MODEL_KEY` in `src/settings.py` sets the CLI default (`xlm-roberta`), but the trained
  checkpoints actually in use are BETO v1/v2.
- Hyperparameters (`lr`, `batch_size`, `grad_accum`, `max_tokens`, `force_fp32`) live in
  each `ModelConfig`, not in `Settings`.
- XLM-RoBERTa was chosen over DeBERTa-v3, which produces NaN gradients even in fp32.
- Related: `docs/ood-signal-provenance.md` for the OOD signals' sources.
