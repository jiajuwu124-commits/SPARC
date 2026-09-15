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
from the 300,000 released prediction records, recompute the 15,000-row
development suite, and regenerate the numerical paper artifacts. This public
tree supplies the code path but does not redistribute those evidence files.

The public tree itself has a small, data-free dynamic component check:

```bash
python scripts/smoke_sparc_current_paper.py
pytest -q
```
