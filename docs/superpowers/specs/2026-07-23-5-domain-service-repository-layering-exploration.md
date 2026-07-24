# Project-Wide Domain/Service/Repository Layering — Exploratory Spec (Not for Implementation)

This is a **reference document for a future decision**, not a plan to execute now. It exists
because implementing the prediction-registry feature (specs 1-4, same date) raised the
question of whether this project should adopt formal domain/service/repository layering
project-wide — a real, legitimate question, but out of scope for one feature to decide
unilaterally. This document audits the current structure, sketches what layering would
look like, and estimates the cost, so the decision is informed whenever it's actually made.
**No touch list, no branch, nothing here should be implemented against this document
directly** — if the project does adopt this later, it gets its own proper spec at that
time, written against the codebase as it exists then.

## Current state: what this codebase actually is

Every cross-cutting concern in this project today is a **plain function module** — no
classes-as-interfaces, no dependency-injection framework, one implementation per concern:

| Module | What it does | Used by |
|---|---|---|
| `src/ood.py` | OOD math (Mahalanobis, cosine, k-NN, TF-IDF stats) | `training/pipeline.py`, `inference/classify.py` |
| `src/embeddings.py` | Model forward-pass → `[CLS]` embeddings | `training/pipeline.py`, `cli/_ood_common.py` |
| `src/svm_reviewer.py` | SVM independent reviewer signal | `training/pipeline.py`, `inference/classify.py` |
| `src/wandb.py` | All W&B interaction | `training/pipeline.py`, CLI `--log-wandb` flags |
| `src/ingestion/` | PDF text extraction (MarkItDown/OCR chain) | `training/`, `inference/` |
| `src/inference/classify.py` | `BertTunningClassifier` — the actual prediction logic | `inference/pipeline.py`, `api/routes/predict/` |
| `src/registry.py` (new, specs 1-4) | SQLite persistence for predictions | `inference/pipeline.py` (CLI), `api/routes/predict/` (API) |

The one place resembling dependency injection is FastAPI's own `Depends()` mechanism —
`_get_clf(request) -> BertTunningClassifier` in `predict/endpoints.py` — which is native
FastAPI, not a project-introduced pattern.

**Why this has worked so far**: this is fundamentally a single-model ML pipeline, not a
multi-team CRUD application. There's one persistence need (until specs 1-4, none at all),
one delivery mechanism split two ways (CLI + a thin FastAPI layer over the same
`predict_pdf`/`predict_text` functions), and every "swap this implementation" question
that has actually come up (BETO vs. XLM-RoBERTa vs. MiniLM) was already solved by the
existing `MODEL_REGISTRY` pattern (`src/training/models/`), not by an abstraction this
document is proposing.

## What layering would look like, if adopted

A conventional domain/service/repository split for this codebase's actual shape:

```
src/
├── domain/              Pure business rules, no I/O, no framework imports
│   ├── prediction.py     PredictionRecord entity, RiskScore/Smell value objects
│   ├── ood_rules.py       is_out_of_distribution() logic (currently src/ood.py)
│   └── review_rules.py    decide_review_route() (currently inference/classify.py)
├── infrastructure/       Adapters to the outside world
│   ├── sqlite_prediction_repository.py   (currently src/registry.py)
│   ├── easyocr_extractor.py               (currently ingestion/extractors/ocr.py)
│   ├── huggingface_model.py               (currently inference/classify.py's model loading)
│   └── wandb_client.py                    (currently src/wandb.py)
├── application/          Services orchestrating domain + infrastructure
│   ├── prediction_service.py   predict_pdf/predict_folder's orchestration
│   └── registry_service.py     the corrected_label validation from spec 4
└── api/ , cli/           Thin delivery layers calling application services
```

Every current module maps to *somewhere* in this structure — this project already has an
implicit domain/infrastructure split, it's just not expressed as directory boundaries or
interfaces.

## Cost estimate

- **Every existing top-level module moves and gets re-homed**: `ood.py`, `embeddings.py`,
  `svm_reviewer.py`, `wandb.py`, `registry.py`, plus reclassifying pieces of
  `inference/classify.py` and `ingestion/extractors/`. Realistically 15-20 files touched.
- **~20+ import statements updated** across `src/`, `tests/`, and `main.py` — every
  `from src.ood import ...`, `from src.svm_reviewer import ...` style import changes.
- **Interfaces/protocols for anything meant to be swappable** — e.g. an
  `ExtractorProtocol`, a `PredictionRepositoryProtocol` — each needs at least one fake/stub
  implementation to make the abstraction pay for itself in tests, or it's ceremony with no
  payoff (the exact judgment call already made explicitly for the registry feature).
- **Full test suite re-run and likely partially rewritten** — tests that currently import
  `from src.ood import compute_class_stats` etc. directly would need updating for new
  module paths regardless of whether test *behavior* changes.
- **A real, multi-session effort** — not a single PR. Best done incrementally (one module
  at a time, mirroring how the SVM/OOD SRP-OCP remediation work earlier in this project's
  history — `docs/superpowers/specs/2026-07-16-svm-signal-srp-ocp-remediation-design.md` —
  was staged rather than done in one sweep), not attempted as one big-bang refactor.

## When this would actually earn its cost

Not a permanent "no" — these are the conditions under which the calculus would flip:

1. **A second persistence backend genuinely needed** — e.g. the registry needs to run
   against Postgres in some deployment and SQLite in another. Today there's exactly one
   backend, one deployment shape (a single Docker container).
2. **A second delivery mechanism beyond CLI + the thin FastAPI wrapper** — e.g. a message-
   queue consumer, a gRPC service — where the *same* application logic needs to be reached
   three+ different ways instead of two.
3. **Multiple people/teams working on different layers simultaneously**, where the
   interface boundary is what lets them work without stepping on each other — currently
   this is effectively a single-maintainer project.
4. **The domain logic itself grows complex enough that testing it through the full stack
   (real SQLite, real model, real FastAPI) becomes slow or awkward** — right now
   `is_out_of_distribution()`/`decide_review_route()` are already pure functions, testable
   in isolation without any layering at all, which is most of what a domain layer would
   buy you, already achieved without the ceremony.

## Recommendation

Don't adopt this now. The current flat-module structure is doing its job — every
"different implementation" need that has actually materialized (three model
architectures, four OOD signals, an SVM reviewer signal) was solved without a formal
layering pattern, via plain modules and a lightweight registry (`MODEL_REGISTRY`). Revisit
if/when one of the four trigger conditions above actually shows up, not preemptively.
