# CantorNet

The code for our NeurReps 2024 workshop paper:

**"CantorNet: A Sandbox for Testing Topological and Geometrical Measures"**  
OpenReview: <https://openreview.net/forum?id=fekgfpKJXi>

## Quick Start

```bash
python main.py --depth 3 --representation B --point 1/3 1/3
```

Example usage:

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

## Citation

```bibtex
@InProceedings{lewandowski2024cantornet,
    author    = {Lewandowski, Michal and Eghbalzadeh, Hamid and Moser, Bernhard A.},
    title     = {CantorNet: A Sandbox For Testing Geometrical and Topological Complexity Measures},
    booktitle = {NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
    series    = {Proceedings of Machine Learning Research},
    year      = {2024}
}
