# Fast Linear Assignment Sorting - FLAS

<img src="../images/teaser.png" width="100%" title="" alt="main_pic"></img>

[FLAS](https://github.com/Visual-Computing/LAS_FLAS/tree/main?tab=readme-ov-file#las-and-flas) allows to sort 2d grids
based on similarity.
This directory contains the python implementation and shows how to use it.

## Features
- Implementation of [FLAS](https://github.com/Visual-Computing/LAS_FLAS/tree/main?tab=readme-ov-file#las-and-flas) and [DPQ](https://github.com/Visual-Computing/LAS_FLAS/tree/main?tab=readme-ov-file#distance-preservation-quality-dpq) (and other metrics to quantify an arrangement)
- Allows frozen fields
- Allows empty fields (holes)
- Deterministic random when seeded
- Installable via PyPI
- Easy to use python interface with numpy support
- Fast implementation in C++

## Installation
```shell
pip install vc_flas
```

### Install from sources
```shell
git clone https://github.com/Visual-Computing/LAS_FLAS.git
cd LAS_FLAS/c_python
pip install .  # or
pip install -e .  # for develop/editable mode
```

## Usage
### Basic Usage
Given that you have N features with D-dimensions, you can sort these features
based on similarity with the following code:

```python
import numpy as np
from vc_flas import Grid, flas

N, D = 241, 3
features = np.random.random((N, D))
grid = Grid.from_features(features)

arrangement = flas(grid, wrap=True, radius_decay=0.99)

sorted_features = arrangement.get_sorted_features()
height, width, dim = sorted_features.shape
assert (height, width, dim) == (16, 16, 3)

# show_image(sorted_features)
```

### Working with Arrangements and Labels
Often you need not only the features sorted, but other objects (like images for example) as well.
See [this example](https://github.com/Visual-Computing/LAS_FLAS/blob/feat/c_python/c_python/examples/using_arrangements.py) for more information on that.

### Creating Grids
There are more ways to initialize grids. See [here](https://github.com/Visual-Computing/LAS_FLAS/blob/feat/c_python/c_python/examples/create_grids.py) for some examples.

### Metrics
Once you have sorted your features, there are some methods to evaluate the
quality of the arrangements.
```python
arrangement = flas(grid)

print(arrangement.get_distance_preservation_quality())
print(arrangement.get_mean_neighbor_distance())
print(arrangement.get_distance_ratio_to_optimum())
```

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
