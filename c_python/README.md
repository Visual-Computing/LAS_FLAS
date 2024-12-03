# Fast Linear Assignment Sorting - FLAS

<img src="../images/teaser.png" width="100%" title="" alt="main_pic"></img>

[FLAS](https://github.com/Visual-Computing/LAS_FLAS/tree/main?tab=readme-ov-file#las-and-flas) allows to sort 2d grids
based on similarity.
This directory contains the python implementation and shows how to use it.

## Features
- Implementation of [FLAS](https://github.com/Visual-Computing/LAS_FLAS/tree/main?tab=readme-ov-file#las-and-flas) and [DPQ](https://github.com/Visual-Computing/LAS_FLAS/tree/main?tab=readme-ov-file#distance-preservation-quality-dpq) (and other metrics to quantify an arrangement)
- Allows frozen fields
- Allows empty fields (holes)
- Installable via PyPI
- Easy to use python interface with numpy support
- Fast implementation in C++

## Installation
TODO: Install via pypi

## Usage
Given that you have N features with D-dimensions, you can sort these features
based on similarity with the following code:

```python
import numpy as np
from vc_flas import Grid, flas

N, D = 241, 7
features = np.random.random((N, D))
grid = Grid.from_features(features)

arrangement = flas(grid, wrap=True, radius_decay=0.99)

sorted_features = arrangement.get_sorted_features()
height, width, dim = sorted_features.shape
assert (height, width, dim) == (16, 16, 7)
```

## TODO
- Explain arrangements / labels
- How to build grids (GridBuilder, Grid.from_features(), Grid.from_grid_features())
- explain flas() Parameters
- explain metrics

For more usage examples see [examples]() TODO.

## About
**Kai Barthel, Nico Hezel, Klaus Jung, Bruno Schilling and Konstantin Schall**

**HTW Berlin, Visual Computing Group, Germany**

[Visual Computing Group](https://visual-computing.com/)

This is an example implementation of the algorithms from the paper 

***Improved Evaluation and Generation of Grid Layouts using Distance Preservation Quality and Linear Assignment Sorting*** 

Published in COMPUTER GRAPHICS Forum: ([https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14718](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14718))

### Reference

Reference to cite when you use any of the presented algorithms in a research paper:
```
@article{https://doi.org/10.1111/cgf.14718,
    author = {Barthel, K. U. and Hezel, N. and Jung, K. and Schall, K.},
    title = {Improved Evaluation and Generation Of Grid Layouts Using Distance Preservation Quality and Linear Assignment Sorting},
    journal = {Computer Graphics Forum},
    volume = {42},
    number = {1},
    pages = {261-276},
    keywords = {interaction, user studies, visualization, information visualization, high dimensional sorting, assistive interfaces},
    doi = {https://doi.org/10.1111/cgf.14718},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14718},
    year = {2023}
}
```

