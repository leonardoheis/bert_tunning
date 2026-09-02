# OOD & Independent Signal Provenance

Where each signal in `bert_tunning` came from, and which paper backs it. Only signal #6
is cited in the repo today — everything else is either uncited or has a candidate paper
already sitting in the Etapa 2 bibliography that was never linked.

| # | Signal | Code | Origin recorded in repo | Citation | URL |
|---|---|---|---|---|---|
| 1 | Mahalanobis (empirical p-value; χ² informational) | `src/ood.py` | None — motivated by a payment receipt forced into `decreto` | **Proposed:** Lee, K., Lee, K., Lee, H., & Shin, J. (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS | https://arxiv.org/abs/1807.03888 |
| 2 | Cosine z-score to nearest centroid | `src/ood.py` | None — empirical variant of #1 | **Proposed:** Techapanurak, E., Suganuma, M., & Okatani, T. (2020). *Hyperparameter-Free Out-of-Distribution Detection Using Cosine Similarity.* ACCV | https://arxiv.org/abs/1905.10628 |
| 3 | k-NN class-conditional distance | `src/ood.py` | None — added because `otro` broke the one-centroid-per-class assumption, inspired from AI diplomatura content | **Proposed:** Sun, Y., Ming, Y., Zhu, X., & Li, Y. (2022). *Out-of-Distribution Detection with Deep Nearest Neighbors.* ICML | https://arxiv.org/abs/2204.06507 |
| 4 | TF-IDF cosine-centroid | `src/ood.py` | None — built for foreign municipalities, documented as having failed at that | **In Etapa 2:** Schwaar, S., Diez, F., Trebing, M., & Witznick, N. (2025). *Study on Text Classification for Public Administration.* | https://arxiv.org/pdf/2504.09111 |
| 5 | `detect_foreign_municipality` (regex) | `src/ingestion/_text.py` | None — own fix after #4 failed | **In Etapa 2:** Peña, A. et al. (2023). *Leveraging LLMs for Topic Classification in the Domain of Public Affairs.* (expert regex annotation layer) | https://doi.org/10.1007/978-3-031-41498-5_2 · https://arxiv.org/abs/2306.02864 |
| 6 | SVM one-vs-rest reviewer | `src/svm_reviewer.py` | **Cited** — `docs/superpowers/specs/2026-07-15-svm-independent-reviewer-design.md:10` | **In Etapa 2:** Peña et al. (2023) — RBF + `class_weight="balanced"` (Weighted Cross Entropy Loss), no PCA | https://doi.org/10.1007/978-3-031-41498-5_2 · https://arxiv.org/abs/2306.02864 |
| 7 | SVM/softmax disagreement | `src/inference/classify.py` | None — derived from #6 | **In Etapa 2 (partial):** Peña et al. (2023), one-vs-all decoupled detectors | https://doi.org/10.1007/978-3-031-41498-5_2 · https://arxiv.org/abs/2306.02864 |
| 8 | Confidence gate → `human_review` | `decide_review_route()` | None | **In Etapa 2:** Gosmar, D., & Zenezini, G. (2026). *MADP: A Multi-Agent Pipeline for Sustainable Document Processing with Human-in-the-Loop.* (Validator agent) | https://arxiv.org/abs/2605.17159 |

## Notes

- **#1–#3 are the gap.** The actual OOD math has no paper in the Etapa 2 bibliography.
  Three new references would need adding if the final report cites them.
- **#4 — cite the limitation, not the success.** Schwaar et al.'s own stated weakness
  ("bag-of-words ignora el orden de las palabras y la semántica contextual") is the
  published explanation for why our TF-IDF signal got diluted by shared legal boilerplate
  and missed the cross-jurisdiction case it was built for.
- **#8 — a genuine delta from MADP.** MADP's gate fires on *low confidence*; ours fires on
  OOD or classifier disagreement **regardless of confidence**. Etapa 2 §5 already flags
  free-text confidence estimation as unsolved by any reviewed antecedent — signals #1–#7
  are this project's answer to that risk.
