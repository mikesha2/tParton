"""QCD constants and parameters used in PDF evolution.

This module defines the color factors, flavor factors, and beta function
coefficients used in the DGLAP evolution equations.

Navigation
----------
- Home: https://mikesha2.github.io/tParton/
- Examples & Tutorials: https://mikesha2.github.io/tParton/examples.html
- API Documentation: https://mikesha2.github.io/tParton/api/tparton/

References
----------
- Sha, C.M. & Ma, B. (2025). arXiv:2409.00221
"""

import numpy as np

def constants(CG, n_f):
    """Compute QCD constants in terms of number of colors and flavors.
    
    Calculates the color factors (NC, CF), flavor factor (Tf), and
    QCD beta function coefficients (β₀, β₁) used throughout the evolution.
    
    Args:
        CG:
        Number of colors, NC (typically 3 for QCD)
        n_f:
        Number of active quark flavors (typically 3-6 depending on Q²)
    
    Returns:
    tuple of float
        (NC, CF, Tf, beta0, beta1) where:
        
        - NC : Number of colors (= CG)
        - CF : Fundamental Casimir operator = (NC² - 1)/(2NC)
        - Tf : Flavor factor = TR × n_f, where TR = 1/2
        - beta0 : Leading QCD beta function coefficient
        - beta1 : Next-to-leading QCD beta function coefficient
    
    Note:
    These constants are defined after Eq. (4) in the paper.
    
    The beta function coefficients govern the running of αs:
    
    - β₀ = (11/3)NC - (4/3)TR·n_f
    - β₁ = (34/3)NC² - (10/3)NC·n_f - 2CF·n_f
    
    For standard QCD with NC=3:
    
    - CF = 4/3
    - beta0 ≈ 11 - (4/3)n_f
    
    Example:
    >>> from tparton.constants import constants
    >>> NC, CF, Tf, beta0, beta1 = constants(CG=3, n_f=5)
    >>> print(f"CF = {CF:.4f}, beta0 = {beta0:.4f}")
    CF = 1.3333, beta0 = 7.3333
    """
    NC = CG
    CF = (NC * NC - 1) / NC / 2
    TR = 1/2
    Tf = TR * n_f
    beta0 = 11 / 3 * CG - 4 / 3 * TR * n_f
    beta1 = 34 / 3 * CG ** 2 - 10 / 3 * CG * n_f - 2 * CF * n_f
    return NC, CF, Tf, beta0, beta1

def split_pdf_input(pdf):
    """Normalize a user-supplied PDF into an (x grid, x*pdf(x) values) pair.

    This is the single canonical input-handling path shared by both evolution
    methods, so that every documented input format is accepted identically and
    both methods always work with plain 1D float arrays downstream.

    Parameters
    ----------
    pdf : array_like
        The input PDF in the tilde convention x*f(x), in any of three formats:

        - 1D, shape ``(N,)``: the x*f(x) values alone, for which an x grid
          linearly spaced on [0, 1] is assumed.
        - 2D single column, shape ``(N, 1)``: the same as the 1D case.
        - 2D two column, shape ``(N, 2)``: explicit
          ``[[x0, x0*f(x0)], [x1, x1*f(x1)], ...]`` pairs. Note that the second
          column is x*f(x), not f(x).

    Returns
    -------
    xs : ndarray
        1D array of x values, ascending.
    values : ndarray
        1D array of x*f(x) values at `xs`.

    Raises
    ------
    ValueError
        If `pdf` is not one of the three formats above.

    Example
    -------
    >>> import numpy as np
    >>> from tparton.constants import split_pdf_input
    >>> xs, vals = split_pdf_input(np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]]))
    >>> xs
    array([0. , 0.5, 1. ])
    """
    pdf = np.asarray(pdf, dtype=float)
    if pdf.ndim == 1 or (pdf.ndim == 2 and pdf.shape[-1] == 1):
        # Only the x*pdf(x) values were supplied, so assume a linear spacing of
        # x from 0 to 1 inclusive.
        values = pdf.reshape(-1)
        xs = np.linspace(0, 1, len(values))
    elif pdf.ndim == 2 and pdf.shape[-1] == 2:
        # Otherwise split the (x, x*pdf(x)) pairs into separate 1D arrays.
        xs, values = pdf[:, 0].copy(), pdf[:, 1].copy()
    else:
        raise ValueError(
            'pdf must have shape (N,), (N, 1) or (N, 2) holding x*pdf(x) '
            f'values; got shape {pdf.shape}'
        )
    return xs, values
