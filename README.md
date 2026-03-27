# CantorNet

Compact reference code for constructing the CantorNet decision boundary and its equivalent ReLU representations from the NeurReps 2024 workshop paper:

**"CantorNet: A Sandbox for Testing Topological and Geometrical Measures"**  
OpenReview: <https://openreview.net/forum?id=fekgfpKJXi>

The repository is intentionally small. Its goal is to expose the core math cleanly:

- exact finite Cantor-set approximations using `Fraction`
- piecewise-linear CantorNet boundary vertices and hyperplanes
- the compact recursion-based ReLU construction (representation A)
- alternative DNF-like ReLU constructions (representations B and C)

## Quick Start

```bash
python main.py --depth 3 --representation B --point 1/3 1/3
```

Example programmatic usage:

```python
from fractions import Fraction

import numpy as np

from cantornet import (
    dnf_representation_weights,
    decision_boundary_vertices,
    recursive_representation_weights,
)

vertices = decision_boundary_vertices(3)
recursive = recursive_representation_weights(3)
dnf = dnf_representation_weights(
    3,
    np.array([Fraction(1, 3), Fraction(1, 3)], dtype=object),
    representation="B",
)
```

## Notes

- The implementation prefers exact rational arithmetic where that helps keep the geometry interpretable.
- `main.py` is a lightweight inspection CLI rather than a training script.
- The code is aimed at research demos and reproducible constructions, not a general deep-learning framework.

## Citation

```bibtex
@InProceedings{lewandowski2024cantornet,
    author    = {Lewandowski, Michal and Eghbal-zadeh, Hamid and Moser, Bernhard A.},
    title     = {CantorNet: A Sandbox For Testing Geometrical and Topological Complexity Measures},
    booktitle = {NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
    series    = {Proceedings of Machine Learning Research},
    year      = {2024}
}
