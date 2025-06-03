# tParton

Copyright 2025 Congzhou M Sha

This is an evolution code for the transversity parton distribution functions encountered in hadronic physics. The associated preprint can be found at [https://arxiv.org/abs/2409.00221v2](https://arxiv.org/abs/2409.00221v2). If you found this work helpful, please cite our preprint, and eventually our published paper!

## Installation

First, ensure that you have a Python package system installed. We recommend Miniconda [https://www.anaconda.com/docs/getting-started/miniconda/install#anaconda-website](https://www.anaconda.com/docs/getting-started/miniconda/install#anaconda-website).

In your terminal, run:
```
conda create -n NAME -c conda-forge jupyterlab pip -y
conda activate NAME
pip install tparton
```

`NAME` is your choice of label for the conda environment you are creating. The above commands only need to be executed once. To use this environment again after a terminal or computer restart, execute `conda activate {NAME}` again.

## Running tParton as a standalone script

To evolve a transversity pdf, run the command `python -m tparton m`. This uses the Mellin moment by default. To use the energy scale integration method, run the command `python -m tparton t`. Use `python -m tparton m -h` or `python -m tparton t -h` for help.

An example run will look like:

`python -m tparton m INPUTPDF.dat 3.1 10.6 --morp plus -o OUTPUTPDF.dat`

We are evolving the `INPUTPDF.dat` from 3.1 to 10.6 GeV<sup>2</sup>, for the plus type distribution, and with output written to `OUTPUTPDF.dat`.

## Importing the tParton package

To use the Hirai ODE integration method:
`from tparton.t_evolution import evolve`

To use the Vogelsang Mellin moment method:
`from tparton.m_evolution import evolve`

Run `help(evolve)` for a description of the inputs.
