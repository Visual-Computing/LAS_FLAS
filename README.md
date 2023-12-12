# LAS/FLAS & DPQ

**Kai Barthel, Nico Hezel, Klaus Jung, and Konstantin Schall**

**HTW Berlin, Visual Computing Group, Germany**

[Visual Computing Group](https://visual-computing.com/)

This is an example implementation of the algorithms from the paper 

***Improved Evaluation and Generation of Grid Layouts using Distance Preservation Quality and Linear Assignment Sorting*** 

Published in COMPUTER GRAPHICS Forum: ([https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14718](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14718))


<img src="images/teaser.png" width="100%" title="" alt="main_pic"></img>

### Abstract

Images sorted by similarity enables more images to be viewed simultaneously, and can be very useful for stock photo agencies or e-commerce applications. Visually sorted grid layouts attempt to arrange images so that their proximity on the grid corresponds as closely as possible to their similarity. Various metrics exist for evaluating such arrangements, but there is low experimental evidence on correlation between human perceived quality and metric value. We propose Distance Preservation Quality (DPQ) as a new metric to evaluate the quality of an arrangement. Extensive user testing revealed stronger correlation of DPQ with user-perceived quality and performance in image retrieval tasks compared to other metrics.
In addition, we introduce Fast Linear Assignment Sorting (FLAS) as a new algorithm for creating visually sorted grid layouts. FLAS achieves very good sorting qualities while improving run time and computational resources.

### Distance Preservation Quality (DPQ)

<div align="center">
    <img src="images/delta_D_plot.png" width="60%" title="" alt="main_pic"></img>
</div>


<div align="center">
    <img src="images/DPQ_eq.png" width="60%" title="" alt="main_pic"></img>
</div>

We define the Distance Preservation Gain vector ∆D as the difference between the average neighborhood distances of a random arrangement and a sorted arrangement S. The final Distance Preservation Quality DPQ(S)_p is then the ratio between the p-norms of ∆D of the given sorted arrangement S and that of a theoretical optimal arrangement, where all distances are perfectly preserved.

### LAS and FLAS

Both algorithmns sort a given number of vectors on a 2-dimensional grid using a linear-assignement solver. 

LAS (Linear Assignment Sorting) is a simple algorithm that uses all vectors at each step, which results in good arrangements, but becomes quite slow for larger sets of vectors. 

Fast Linear Assignments Sorting (FLAS) is able to handle larger quantities of vectors by replacing the global assignment with multiple local swaps. This approach allows much faster sorting while having little impact on the quality of the arrangement.


<p float="top" align="middle">
    <img src="images/LAS_algo.png" width="45%" title="" alt="main_pic" align="top" margin="1000px"></img>
    &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
    <img src="images/FLAS_algo.png" width="45%" title="" alt="main_pic"></img>
</p>

### Example

We provide a Jupyter notebook with full python code for both sorting algorithms and the quality measure DPQ:

[DPQ_LAS&FLAS](python/LAS_FLAS_DPQ_colors.ipynb)

Please install the python packages listed in [python/requirements.txt](python/requirements.txt) before executing the notebook on your local machine

```bash
pip install -r requirements.txt
```

*Update:*
We added the images and feature vectors from the sets described in the paper. You can try them out in this [notebook](LAS_FLAS_DPQ_images.ipynb). Please unzip the data.zip file before running this notebook.

## Reference

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