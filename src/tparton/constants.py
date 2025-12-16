"""QCD constants and parameters used in PDF evolution.

This module defines the color factors, flavor factors, and beta function
coefficients used in the DGLAP evolution equations.

References
----------
.. [1] Sha, C.M. & Ma, B. (2024). arXiv:2409.00221
"""

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