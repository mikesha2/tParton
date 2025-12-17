# Copyright Congzhou M Sha 2024
"""Direct integration method for transversity PDF evolution (Hirai method)

This module implements the Hirai et al. method for evolving transversity parton
distribution functions by directly solving the DGLAP integro-differential equation.

The evolution is performed by:
1. Discretizing the momentum fraction x and energy scale Q² into grids
2. Using Euler method for Q² evolution steps
3. Using Simpson's rule for integration over x

This method provides direct control over discretization but can be computationally
expensive for fine grids. It is more suitable when precise control over numerical
parameters is needed.

Key Functions
-------------
evolve : Main function to evolve a transversity PDF
splitting : Compute the transversity splitting functions
alp2pi : Compute the strong coupling α_s(Q²)/(2π)

Theoretical Background
---------------------
Solves the DGLAP equation (Eq. 1 from [1]):

    ∂/∂t Δ_T q^±(x,t) = (α_s(t)/(2π)) Δ_T P_q^±(x) ⊗ Δ_T q^±(x,t)

where t = ln(Q²), Q² is the energy scale, and ⊗ denotes Mellin convolution
(Eq. 3). The tilde notation f̃(x) = x·f(x) is used throughout (Eq. 2).

The NLO strong coupling is given by Eq. (4):

    α_s^{NLO}(Q²) = (4π)/(β₀ ln(Q²/Λ²)) [1 - (β₁ ln(ln(Q²/Λ²)))/(β₀² ln(Q²/Λ²))]

The splitting function has LO (Eq. 9) and NLO (Eqs. 11-12) contributions,
with plus distribution regularization (Eq. 10).

References
----------
- Hirai, M., Kumano, S., & Miyama, M. (1998). Comput. Phys. Commun. 111, 150-166
- Sha, C.M. & Ma, B. (2025). arXiv:2409.00221
"""
__docformat__ = "numpy"

from .constants import constants
import numpy as np
from scipy.integrate import simpson
from numpy._core.multiarray import interp
from scipy.integrate import odeint
from scipy.special import spence

pi = np.pi

def alp2pi(t: float, lnlam: float, order: int, beta0: float, beta1: float) -> float:
    """Compute α_s(Q²)/(2π) using the analytical approximation.
    
    Implements Eq. (4) from the paper, normalized by 2π:
    
        α_s^{NLO}(Q²)/(2π) = (2)/(β₀ ln(Q²/Λ²)) [1 - (β₁ ln(ln(Q²/Λ²)))/(β₀² ln(Q²/Λ²))]
    
    where t = ln(Q²) and lnlam = 2 ln(Λ). This normalization is convenient
    for the DGLAP evolution equation.
    """
    dlnq2 = t - lnlam
    alpha = 4 * pi / beta0 / dlnq2
    alpha_factor = 1 if order == 1 else (1 - beta1 * np.log(dlnq2) / beta0**2 / dlnq2)
    alpha_factor /= (2 * pi)
    return alpha_factor * alpha

def splitting(z: np.ndarray, CF: float, order: int, sign: int, CG: int, Tf: float):
    """Compute transversity splitting functions at given momentum fraction.
    
    Evaluates the full NLO splitting function Eq. (8):
    
        Δ_T P_q^±(x) = Δ_T P_qq^{(0)}(x) + (α_s(Q²))/(2π) Δ_T P_q^{(1)}(x)
    
    The LO term (Eq. 9) is:
    
        Δ_T P_qq^{(0)}(x) = C_F [(2x)/((1-x)_+) + (3/2)δ(1-x)]
    
    The NLO contribution (Eq. 11) splits into regular (Eq. 12), plus (Eq. 12),
    and delta function (Eq. 13) parts. Plus distributions are defined by Eq. (10).
    
    Returns
    -------
    tuple
        (p0, p1, p0pf, p1pf, plus0, del0, plus1, del1) where:
        
        - p0, p1: Regular function parts (LO and NLO)
        - p0pf, p1pf: Plus function coefficients  
        - plus0, del0: LO plus and delta function contributions
        - plus1, del1: NLO plus and delta function contributions
    
    Notes
    -----
    Uses Spence function S(x) from scipy.special (Eq. 15). The Spence function
    is related to the dilogarithm Li₂(z) by S(x) = -Li₂(1-x).
    Regularizes singularities at z=1 with epsilon = 1e-100.
    
    See Also
    --------
    integrate : Uses splitting functions for PDF convolution
    """

    # p0 and p0pf correspond to the first term in Eq. (9) containing a plus function prescription
    p0 = CF * 2 * z / (1 - z+1e-100)
    p0pf = -CF * 2 / (1 - z+1e-100)

    # Define z1 and z2 for vectorization of further computations below
    z1 = 1 / (1 + z)
    z2 = z / (1 + z)
    dln1 = np.log(z1)
    dln2 = np.log(z2)
    # The SciPy convention for the Spence function (aka the dilogarithm) differs from that of Hirai, as discussed in Eq. (16)
    s2 = -spence(z2) + spence(z1) - (dln1 ** 2 - dln2 ** 2) * 0.5

    # 1-z
    omz = 1 - z
    #######################################################################################################
    # Non-plus function, non-delta function contributions to the splitting function
    #######################################################################################################
    # First, we evaluate the plus function contributions
    if order == 2:
        p1 = nlo_regular_terms(z, omz, s2, CF, CG, Tf, sign)
    else:
        p1 = 0
    
    #######################################################################################################
    # Plus function, f(x) g(x)_+ in Eq. (10), contributions to the splitting function
    #######################################################################################################
    # Enforcing the plus prescriptions for the relevant terms in Eq. (12)
    # The plus prescription in line 0 of Eq. (12) results in 0, so it is ignored
    # Line 2 plus prescription in Eq. (12)
    p1pf = nlo_plus_terms(omz, CF, CG, Tf)

    # Necessary to avoid numerical singularities
    p0[-1] = 0
    p0pf[-1] = 0
    if order == 2:
        p1[-1] = 0
        p1pf[-1] = 0

    # The LO plus and delta function contributions to the integrals
    plus0 = CF * 2
    del0 = CF * 3 / 2

    #######################################################################################################
    # Plus function, -f(1) g(x)_+ in Eq. (10), and delta function contributions to the splitting function
    #######################################################################################################
    if order == 2:
        plus1, del1 = nlo_plus_delta_constants(CF, CG, Tf)
    else:
        plus1, del1 = 0, 0

    return p0, p1, p0pf, p1pf, plus0, del0, plus1, del1

def nlo_regular_terms(z: np.ndarray, omz: np.ndarray, s2: np.ndarray,
                       CF: float, CG: int, Tf: float, sign: int) -> np.ndarray:
    """NLO non-plus, non-delta contributions to ΔT P_qq(z).

    Implements the regular parts of Eq. (12) and the additional term Eq. (13).

    Parameters
    ----------
    z : ndarray
        Momentum fraction grid.
    omz : ndarray
        1 - z values.
    s2 : ndarray
        Dilogarithm combination from Eq. (16).
    CF, CG, Tf : float
        Color and flavor factors.
    sign : int
        +1 for plus, -1 for minus distribution.

    Returns
    -------
    ndarray
        Regular NLO contribution array.
    """
    lnz = np.log(z)
    lno = np.log(omz+1e-100)
    dP0 = 2 * z / (omz+1e-100)
    pp1 = omz - (3 / 2 + 2 * lno) * lnz * dP0
    pp2 = -omz + (67/9 + 11/3 * lnz + lnz**2 - pi**2 / 3) * dP0
    pp3 = (-lnz - 5/3) * dP0
    dpqq = CF * CF * pp1 + CF * CG * 0.5 * pp2 + 2 / 3 * CF * Tf * pp3
    pp4 = -omz + 2 * s2 * 2 * -z / (1 + z)
    dpqqb = CF * (CF - CG / 2) * pp4
    p1 = dpqq + sign * dpqqb
    p1[0] = 0
    return p1

def nlo_plus_terms(omz: np.ndarray, CF: float, CG: int, Tf: float) -> np.ndarray:
    """NLO plus-prescription function coefficients.

    Parameters
    ----------
    omz : ndarray
        1 - z values.
    CF, CG, Tf : float
        Color and flavor factors.

    Returns
    -------
    ndarray
        Plus-function coefficients p1pf.
    """
    p2plus = -(67/9 - pi**2/3) * 2 / (omz+1e-100)
    p3plus = 5 / 3 * 2 / (omz+1e-100)
    return CF * CG / 2 * p2plus + 2 / 3 * CF * Tf * p3plus

def nlo_plus_delta_constants(CF: float, CG: int, Tf: float) -> tuple[float, float]:
    """NLO plus and delta-function constants evaluated at z → 1.

    Returns
    -------
    tuple
        (plus1, del1) constants in Eq. (12).
    """
    zta = 1.2020569031595943
    del1 = CF * CF * (3 / 8 - pi**2 / 2 + 6 * zta) + \
        CF * CG / 2 * (17 / 12 + 11 * pi**2 / 9 - 6 * zta) - \
        2 / 3 * CF * Tf * (1 / 4 + pi**2 / 3)
    p2pl = (67 / 9 - pi**2/3) * 2
    p3pl = -5 / 3 * 2
    plus1 = CF * CG / 2 * p2pl + 2 / 3 * CF * Tf * p3pl
    return plus1, del1

# Define the integration step required here
def integrate(pdf: np.ndarray, i: int, z: np.ndarray, alp: float, order: int, 
              CF: float, sign: int, CG: int, Tf: float, xs: np.ndarray) -> float:
    """Perform convolution of PDF with splitting function at a given x value.
    
    Implements the Mellin convolution integral from Eq. (19).
    Uses Simpson's rule for numerical integration. Handles plus distribution
    prescriptions via ln(1-x) terms. Interpolates PDF between grid points
    for smooth convolution.
    """

    # Handle the base case of an empty array
    if len(z) == 0:
        return 0
    
    # Evaluate the splitting function at the points z
    p0, p1, p0pf, p1pf, plus0, del0, plus1, del1 = splitting(z, CF, order, sign, CG, Tf)

    # Implement Eq. (19), instead of Eq. (7) for the convolution
    func = ((p0 + (alp * p1 if order == 2 else 0)) * interp(xs[i] / z, xs, pdf)) + \
        (p0pf + (alp * p1pf if order == 2 else 0)) * pdf[i]

    # When handling the plus prescription, there is a common factor of ln(1-x) when integrating Eq. (10)
    lno = plus_log_term(xs[i])
    estimate = simpson(func, x=z) + (plus0 * lno + del0) * pdf[i]
    if order == 2:
        estimate += alp * (plus1 * lno + del1) * pdf[i]

    return estimate

def z_grid(xi: float, n_z: int, logScale: bool) -> np.ndarray:
    """Construct z-grid for convolution integral.

    Parameters
    ----------
    xi : float
        Current x value. Grid spans from `xi` to 1.
    n_z : int
        Number of z points (inclusive of endpoints).
    logScale : bool
        Use logarithmic spacing near 1 if True; otherwise linear spacing.

    Returns
    -------
    ndarray
        z-grid array of shape `(n_z + 1,)` from `xi` to `1`.
    """
    if logScale:
        # Log-spacing concentrated near 1 for peaked PDFs
        return np.power(10, np.linspace(np.log10(max(xi, 1e-15)), 0, n_z + 1))
    return np.linspace(xi, 1, n_z + 1)

def plus_log_term(x: float) -> float:
    """Compute the common ln(1-x) factor from plus-prescription integrals.

    Parameters
    ----------
    x : float
        The current `x` value in the convolution.

    Returns
    -------
    float
        `ln(1-x)` stabilized for `x → 1` to avoid `log(0)`.
    """
    return np.log(max(1 - x, 1e-100))

def evolve(
    pdf: np.ndarray,
    Q0_2: float = 0.16,
    Q2: float = 5.0,
    l_QCD: float = 0.25,
    n_f: int = 5,
    CG: float = 3,
    n_t: int = 100,
    n_z: int = 500,
    morp: str = 'plus',
    order: int = 2,
    logScale: bool = False,
    verbose: bool = False,
    Q0_2_a: float = 91.1876 ** 2,
    a0: float = 0.118 / 4 / np.pi,
    alpha_num: bool = True
) -> np.ndarray:
    """Evolve transversity PDF using the direct integration method (Hirai).
    
    This is the main function for PDF evolution using direct numerical integration
    of the DGLAP equation (Eq. 1). More robust for peaked PDFs but slower than
    Mellin method.
    
    This method:
    1. Discretizes t = ln(Q²) into n_t Euler steps
    2. At each step, computes the convolution integral (Eq. 19):
       
           f̃(x) ⊗ g(x) = ∫ dx̃ f̃(x/z) g(z)
       
       using Simpson's rule with n_z integration points
    3. Updates PDF using forward Euler: f̃(t + dt) ≈ f̃(t) + dt·f̃'(t)
    
    Parameters
    ----------
    pdf : ndarray
        Input PDF as x*f(x). Can be 1D array (values at x evenly
        spaced on [0, 1]) or 2D array ([[x0, x0*f(x0)], [x1, x1*f(x1)], ...]).
    Q0_2 : float, optional
        Initial energy scale squared in GeV² (default: 0.16).
    Q2 : float, optional
        Final energy scale squared in GeV² (default: 5.0).
    l_QCD : float, optional
        QCD scale parameter Λ in GeV (default: 0.25).
        Only used if alpha_num=False.
    n_f : int, optional
        Number of active quark flavors (default: 5).
    CG : float, optional
        Number of colors, NC (default: 3).
    n_t : int, optional
        Number of Euler time steps (default: 100).
        More steps = better accuracy but slower.
    n_z : int, optional
        Number of z points for convolution integrals (default: 500).
        More points = better accuracy but slower.
    morp : str, optional
        Distribution type (default: 'plus'). Options are 'plus'
        (ΔT q⁺ = ΔT u + ΔT d) or 'minus' (ΔT q⁻ = ΔT u - ΔT d).
    order : int, optional
        Perturbative order (default: 2). Use 1 for LO or 2 for NLO.
    logScale : bool, optional
        Use logarithmic spacing for z points (default: False).
        Recommended for peaked PDFs.
    verbose : bool, optional
        Print progress (time step count) if True (default: False).
    Q0_2_a : float, optional
        Reference scale Q₀² where αs is known, in GeV² (default: 91.1876²).
        Only used if alpha_num=True.
    a0 : float, optional
        Reference coupling αs(Q0_2_a)/(4π) (default: 0.118/(4π)).
        Only used if alpha_num=True.
    alpha_num : bool, optional
        Use numerical ODE evolution for αs if True (default: True).
        If False, uses analytical approximation.
    
    Returns
    -------
    ndarray
        Evolved PDF as 2D array [[x, x*f_evolved(x)], ...]. Shape: (n+1, 2)
        where n is the number of input points.
    
    Examples
    --------
    >>> import numpy as np
    >>> from tparton.t_evolution import evolve
    >>> x = np.linspace(0, 1, 100)
    >>> pdf_in = x * (1-x)**3  # x*f(x) format
    >>> pdf_out = evolve(pdf_in, Q0_2=4.0, Q2=100.0, n_t=200, n_z=1000)
    >>> x_out, xf_out = pdf_out[:, 0], pdf_out[:, 1]
    
    See Also
    --------
    m_evolution.evolve : Mellin moment method (Vogelsang)
    integrate : Performs convolution at a single x value
    splitting : Evaluates splitting functions
    
    References
    ----------
    - Hirai, M., Kumano, S., & Saito, N. (1998). Comput. Phys. Commun. 111, 150-160
    - Sha, C.M. & Ma, B. (2025). arXiv:2409.00221
    """
    if pdf.shape[-1] == 1:
        # If only the x*pdf(x) values are supplied, assume a linear spacing from 0 to 1
        xs = np.linspace(0, 1, len(pdf))
    else:
        # Otherwise split the input array
        xs, pdf = pdf[:, 0], pdf[:, 1]

    sign = 1 if morp == 'plus' else -1
    lnlam = 2 * np.log(l_QCD)

    # Calculate the color constants
    _, CF, Tf, beta0, beta1 = constants(CG, n_f)

    # Define the (log) starting and ending energy scales squared
    tmin = np.log(Q0_2)
    tmax = np.log(Q2)

    # Define the timepoints between those energy scales at which Eq. (1) will be integrated
    ts = np.linspace(tmin, tmax, n_t)
    
    def _beta_ode(x, a):
        """QCD beta-function ODE for running coupling.

        Parameters
        ----------
        x : float
            Log-energy variable (t = ln Q²). Not used explicitly.
        a : float
            Coupling `a = α_s/(4π)` at scale `x`.

        Returns
        -------
        float
            Time derivative da/dt according to LO/NLO beta function.
        """
        return -beta0 * a * a - (beta1 * a * a * a if order == 2 else 0)
    ode = _beta_ode

    # SciPy requires that the times be monotonically increasing or decreasing
    less = ts < np.log(Q0_2_a)
    ts_less = ts[less]
    ts_greater =ts[~less]

    # For energies strictly below the reference energy, evolve toward lower t
    alp2pi_num_less = odeint(ode, a0, [np.log(Q0_2_a)] + list(ts_less[::-1]), tfirst=True).flatten() * 2
    alp2pi_num_less = alp2pi_num_less[-1:0:-1]
    # For energies strictly above the reference energy, evolve toward higher t
    alp2pi_num_greater = odeint(ode, a0, [np.log(Q0_2_a)] + list(ts_greater), tfirst=True).flatten() * 2
    # Combine the alpha / 2 pi in increasing order of energy scale
    alp2pi_num_greater = alp2pi_num_greater[1:]
    alp2pi_num = list(alp2pi_num_less) + list(alp2pi_num_greater)
    if alpha_num:
        # Use the numerically evolved alpha_S / 2 pi
        alp2pi_use = alp2pi_num
    else:
        # Use the approximate analytical expression for alpha_S in Eq. (4)
        alp2pi_use = alp2pi(ts, lnlam, order, beta0, beta1)

    # Euler integration of Eq. (1) by a small timestep dt
    dt = (tmax - tmin) / n_t
    res = np.copy(pdf)

    for i, alp in enumerate(alp2pi_use):
        if verbose:
            print(i+1, ' of ', len(ts), 'time steps')
        # Perform the convolution at each x using (possibly log-scaled) z integration points
        inc = np.array([
            integrate(
                res,
                index,
                z_grid(xs[index], n_z, logScale),
                alp,
                order,
                CF,
                sign,
                CG,
                Tf,
                xs,
            )
            for index in range(1, len(xs) - 1)
        ])
        # Ensure that x*pdf(x) = 0 at x = 0 and x = 1
        inc = np.pad(inc, 1)
        res += dt * inc * alp
    return np.stack((xs, res), axis=1)


def main():
    """Command-line interface for direct integration evolution method.
    
    Parses command-line arguments and runs PDF evolution using Hirai's method.
    This function is called when running `python -m tparton t`.
    
    See Also
    --------
    evolve : Main function for PDF evolution
    """
    import argparse, sys
    parser = argparse.ArgumentParser(description='Evolution of the nonsinglet transversity PDF, according to the DGLAP equation.')
    parser.add_argument('type',action='store',type=str,help='The method you chose')
    parser.add_argument('input', action='store', type=str,
                    help='The CSV file containing (x,x*PDF(x)) pairs on each line. If only a single number on each line, we assume a linear spacing for x between 0 and 1 inclusive')
    parser.add_argument('Q0sq', action='store', type=float, help='The starting energy scale in units of GeV^2')
    parser.add_argument('Qsq', action='store', type=float, help='The ending energy scale in units of GeV^2')
    parser.add_argument('--morp',  action='store', nargs='?', type=str, default='plus', help='The plus vs minus type PDF (default \'plus\')')
    parser.add_argument('-o', action='store', nargs='?', type=str, default='out.dat', help='Output file for the PDF, stored as (x,x*PDF(x)) pairs.')
    parser.add_argument('-l', metavar='l_QCD', nargs='?', action='store', type=float, default=0.25, help='The QCD scale parameter (default 0.25 GeV^2). Only used when --alpha_num is False.')
    parser.add_argument('--nf', metavar='n_f', nargs='?', action='store', type=int, default=5, help='The number of flavors (default 5)')
    parser.add_argument('--nc', metavar='n_c', nargs='?', action='store', type=int, default=3, help='The number of colors (default 3)')
    parser.add_argument('--order', metavar='order', nargs='?', action='store', type=int, default=2, help='1: leading order, 2: NLO DGLAP (default 2)')
    parser.add_argument('--nt', metavar='n_t', nargs='?', action='store', type=int, default=100, help='Number of steps to numerically integrate the DGLAP equations (default 100)')
    parser.add_argument('--nz', metavar='n_z', nargs='?', action='store', type=int, default=1000, help='Number of steps for numerical integration (default 1000)')
    parser.add_argument('--logScale', nargs='?', action='store', type=bool, default=True, help='True if integration should be done on a log scale (default True)')
    parser.add_argument('--delim', nargs='?', action='store', type=str, default=' ', help='Delimiter for data file (default \' \'). If given without an argument, then the delimiter is whitespace (i.e. Mathematica output.)')
    parser.add_argument('--alpha_num', metavar='alpha_num', nargs='?', action='store', type=bool, default=True, help='Set to use the numerical solution for the strong coupling constant, numerically evolved at LO or NLO depending on the --order parameter.')
    parser.add_argument('--Q0sqalpha', metavar='Q0sqalpha', nargs='?', action='store', type=float, default=91.1876**2, help='The reference energy squared at which the strong coupling constant is known. Default is the squared Z boson mass. Use in conjunction with --a0. Only used when --alpha_num is True.')
    parser.add_argument('--a0', metavar='a0', nargs='?', action='store', type=float, default=0.118 / 4 / np.pi, help='The reference value of the strong coupling constant a = alpha / (4 pi) at the corresponding reference energy --Q0sqalpha. Default is 0.118 / (4 pi), at energy Q0sqalpha = Z boson mass squared. Only used when --alpha_num is True.')
    parser.add_argument('-v', nargs='?', action='store', type=bool, default=False, help='Verbose output (default False)')


    args = parser.parse_args()
    f = args.input
    if args.delim is None:
        pdf = np.genfromtxt(f)
        args.delim = ' '
    else:
        pdf = np.genfromtxt(f, delimiter=args.delim)
    Q0sq = args.Q0sq
    Qsq = args.Qsq
    morp = args.morp
    l = args.l
    nf = args.nf
    nc = args.nc
    order = args.order
    nt = args.nt
    nz = args.nz
    logScale = args.logScale
    alpha_num = args.alpha_num
    Q0sqalpha = args.Q0sqalpha
    a0 = args.a0
    verbose = args.v

    res = evolve(pdf,
        Q0_2=Q0sq,
        Q2=Qsq,
        l_QCD=l,
        n_f=nf,
        CG=nc,
        n_t=nt,
        n_z=nz,
        morp=morp,
        order=order,
        logScale=logScale,
        verbose=verbose,
        alpha_num=alpha_num,
        Q0_2_a=Q0sqalpha,
        a0=a0
    )

    np.savetxt(args.o, res.T, delimiter=args.delim)
    if verbose:
        print(res)

if __name__ == '__main__':
    main()
