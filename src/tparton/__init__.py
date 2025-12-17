"""tParton: Evolution of transversity parton distribution functions

tParton is a Python package for evolving transversity PDFs using two complementary methods:

1. **Direct integration (Hirai method)**: Numerically solves the DGLAP equations using 
   discretized grids in x and Q².
   
2. **Mellin moment method (Vogelsang method)**: Uses Mellin transforms and inverse 
   transform via Cohen contour integration for faster, more accurate evolution.

Basic Usage
-----------

Command-line interface::

    # Using Mellin moment method
    python -m tparton m input.dat 3.1 10.6 --morp plus -o output.dat
    
    # Using direct integration method  
    python -m tparton t input.dat 3.1 10.6 --morp plus -o output.dat

Python API::

    from tparton.m_evolution import evolve as m_evolve
    from tparton.t_evolution import evolve as t_evolve
    
    # Evolve using Mellin method (faster)
    result = m_evolve(input_pdf, Q0_squared=3.1, Q_squared=10.6, 
                      morp='plus', order='NLO')
    
    # Evolve using direct integration (more control over discretization)
    result = t_evolve(input_pdf, Q0_squared=3.1, Q_squared=10.6,
                      morp='plus', order='NLO')

See Also
--------
- Full documentation: https://mikesha2.github.io/tParton/
- GitHub repository: https://github.com/mikesha2/tParton
- ArXiv preprint: https://arxiv.org/abs/2409.00221
"""
__docformat__ = "numpy"

from .t_evolution import evolve as t_evolve
from .m_evolution import evolve as m_evolve

__version__ = "1.0.0"
__author__ = "Congzhou M Sha, Bailing Ma"
__all__ = ['t_evolve', 'm_evolve']
