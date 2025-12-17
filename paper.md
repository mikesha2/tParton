---
title: 'tParton: A Python package for next-to-leading order evolution of transversity parton distribution functions'
tags:
  - Python
  - physics
  - particle physics
  - QCD
  - parton distribution functions
  - transversity
  - DGLAP evolution

authors:
  - name: Congzhou M Sha
    orcid: 0000-0001-5301-9459
    corresponding: true
    affiliation: 1
  - name: Bailing Ma
    orcid: 0000-0001-5160-1971
    affiliation: 2
affiliations:
 - name: Penn State College of Medicine, Hershey, PA 17033, USA
   index: 1
 - name: Wake Forest University School of Medicine, Winston-Salem, NC 27101, USA
   index: 2
date: 16 December 2025
bibliography: paper.bib
---

# Summary

Parton distribution functions (PDFs) describe the probability of finding quarks and gluons (collectively called partons) within hadrons such as protons and neutrons. These functions are fundamental to our understanding of quantum chromodynamics (QCD) and are essential for interpreting high-energy physics experiments. The transversity PDF, which encodes information about the transverse spin structure of hadrons, is particularly challenging to measure experimentally and has been less studied computationally compared to unpolarized and helicity PDFs.

`tParton` is a Python package that implements two distinct methods for solving the Dokshitzer–Gribov–Lipatov–Altarelli–Parisi (DGLAP) evolution equations for transversity PDFs at leading order (LO) and next-to-leading order (NLO) in perturbative QCD. The package provides both a command-line interface and a Python API, making it accessible for both quick calculations and integration into larger analysis workflows.

# Statement of need

PDFs must be evolved from one energy scale to another to enable comparisons between different experiments and theoretical predictions. While numerous codes exist for evolving unpolarized and helicity PDFs (such as QCDNUM [@qcdnum], EKO [@EKO], HOPPET [@Salam:2008qg], and APFEL++ [@Bertone:2013vaa; @Bertone:2017gds]), options for transversity PDF evolution are limited. The original Fortran implementation by Hirai et al. [@hirai] is nearly 30 years old and no longer accessible. APFEL++ [@Bertone:2017gds] provides an implementation using direct numerical integration, but no publicly available code has implemented the alternative Mellin moment method proposed by Vogelsang [@Vogelsang97].

`tParton` fills this gap by providing:

1. **Two complementary methods**: A direct integration method (following Hirai et al.) and a Mellin moment method (following Vogelsang), allowing users to choose based on their accuracy and computational needs.

2. **Modern Python implementation**: Built on NumPy [@numpy] and SciPy [@scipy], `tParton` is easy to install via pip and integrates seamlessly with the Python scientific computing ecosystem.

3. **Comprehensive validation**: The package includes extensive examples and validation against both Mathematica implementations and APFEL++ results, with detailed discussion of discretization effects and method comparisons.

4. **Dual interface**: Both command-line tools for standalone use and importable modules for integration into larger projects.

The package is aimed at researchers in hadronic physics, particularly those analyzing semi-inclusive deep inelastic scattering experiments and studying nucleon spin structure. It has been validated and documented in a detailed preprint [@sha2025tparton].

# Implementation

`tParton` implements the DGLAP evolution equation for the transversity PDF:

$$\frac{\partial}{\partial t}\Delta_T q^{\pm}(x,t)=\frac{\alpha_s(t)}{2\pi}\Delta_T P_{q^{\pm}}(x)\otimes\Delta_T q^{\pm}(x,t)$$

where $t=\ln Q^2$, $Q^2$ is the energy scale, $\Delta_T P_{q^{\pm}}$ is the transversity splitting function, and $\otimes$ denotes Mellin convolution.

## Method 1: Direct integration (Hirai method)

The first method discretizes both the momentum fraction $x$ and energy scale $Q^2$ into grids and solves the integro-differential equation using the Euler method for $Q^2$ evolution and Simpson's rule for $x$ integration. This approach is straightforward but can be computationally expensive for fine grids.

## Method 2: Mellin moment method (Vogelsang method)

The second method exploits the convolution theorem for Mellin transforms. The solution is expressed in terms of Mellin moments:

$$\mathcal{M}[\Delta_T q^{\pm}](Q^2;s)=K(s,Q^2,Q_0^2)\mathcal{M}[\Delta_T q^{\pm}](Q_0^2;s)$$

where $K$ contains the evolution kernel depending on the splitting function moments. The evolved PDF is reconstructed via inverse Mellin transform using the Talbot contour integration method. This approach is typically faster and less sensitive to discretization for smooth PDFs.

Both methods support LO and NLO evolution with exact or analytical forms of the running coupling constant $\alpha_s(Q^2)$.

# Examples and validation

The package includes extensive Jupyter notebooks in the `examples/` directory that:

- Generate initial transversity distributions based on literature models [@hirai]
- Compare both evolution methods against each other and against APFEL++
- Demonstrate sensitivity to numerical parameters (grid resolution, timesteps)
- Reproduce figures from the associated preprint [@sha2025tparton]

A separate Mathematica notebook validates the analytical expressions for the Mellin moments of the splitting functions, providing an independent check of the theoretical framework.

Users can evolve a PDF with a single command:

```bash
python -m tparton m input.dat 3.1 10.6 --morp plus -o output.dat
```

Or import and use the package programmatically:

```python
from tparton.m_evolution import evolve
result = evolve(input_pdf, Q0_squared=3.1, Q_squared=10.6, 
                morp='plus', order='NLO')
```

# Acknowledgements

We acknowledge helpful discussions with colleagues in the hadronic physics community and thank the maintainers of APFEL++ for providing comparison benchmarks.

# References
