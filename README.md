# SPARC current-paper code

This repository contains only the code needed by the experiments reported in
the current SPARC manuscript. Superseded experiments and removed paper
artifacts are intentionally excluded.

## Contents

- `src/streammint/`: modules imported by the current router and restoration pipeline.
- `scripts/`: current training, five-backend inference, raw-result audit, and artifact entry points.
- `tests/`: implementation and frozen-protocol checks for those entry points.
- `requirements.txt`: recorded Python dependencies.

Paper files, figures, PowerPoint sources, experiment outputs, datasets, model
checkpoints, credentials, author details, and machine-specific paths are
intentionally excluded from this public code repository.

ImageNet and backend checkpoints must be obtained under their original licenses. The experiment protocol uses seed 1109 and a maximum CUDA allocator fraction of 0.68.

The matching local evidence bundle can run
`scripts/reproduce_sparc_current_paper.py` to reconstruct all 75 Full-15 cells
from 300,000 frozen prediction records, recompute the 15,000-observation
development suite, independently audit the image- and batch-disjoint nested
folds, and regenerate Tables 1--3 and Figures 2--4. Development Fix, Break, and
DeltaAcc are recomputed from compact per-image correctness transitions; they
are not inferred from batch-average utility. This public tree supplies the
code path but does not redistribute evidence, data, checkpoints, or paper
files.

The current numerical path is:

1. `rebuild_sparc_paper_summary.py` and `generate_sparc_full15_table1.py`;
2. `run_sparc_per_image_development_suite.py`;
3. `audit_sparc_per_image_development_suite.py`; and
4. `generate_sparc_final_artifacts.py`.

`build_sparc_per_image_transition_evidence.py` creates the compact transition
bundle from licensed local prediction records. Historical proxy-metric and
removed-paper artifact generators are intentionally absent.

The public tree itself has a small, data-free dynamic component check:

```bash
python scripts/smoke_sparc_current_paper.py
pytest -q
```
