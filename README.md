# tParton

Copyright 2024 Congzhou M Sha

This is an evolution code for the transversity parton distribution functions encountered in hadronic physics.

## Installation

First, install this package using `pip install tparton`.

## Running tParton as a standalone script

To evolve a transversity pdf, run the command `python -m tparton m`. This uses the Mellin moment by default. To use the energy scale integration method, run the command `python -m tparton t`. Use `python -m tparton m -h` or `python -m tparton t -h` for help.

An example run will look like:

`python -m tparton m INPUTPDF.dat 3.1 10.6 --morp plus -o OUTPUTPDF.dat`

We are evolving the `INPUTPDF.dat` from 3.1 to 10.6 GeV<sup>2</sup>, for the plus type distribution, and with output written to `OUTPUTPDF.dat`.

## Importing the tParton package

To use the Hirai ODE integration method:
`from tparton.t_evolution import evolve`

To use the Vogelsang Mellon moment method:
`from tparton.m_evolution import evolve`

Run `help(evolve)` for a description of the inputs.