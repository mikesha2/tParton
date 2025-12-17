# Copyright Congzhou M Sha 2024
"""Mellin moment method for transversity PDF evolution (Vogelsang method)

This module implements the Vogelsang method for evolving transversity parton 
distribution functions using Mellin transforms. The method is typically faster 
and less sensitive to discretization effects compared to direct integration.

The evolution is performed by:
1. Computing Mellin moments of the input PDF
2. Evolving the moments using analytic expressions for splitting function moments
3. Reconstructing the evolved PDF via inverse Mellin transform (Cohen method)

Key Functions
-------------
evolve : Main function to evolve a transversity PDF

Theoretical Background
---------------------
The solution uses the convolution theorem for Mellin transforms:

    M[Δ_T q^±](Q²;s) = K(s,Q²,Q₀²) M[Δ_T q^±](Q₀²;s)

where K contains the evolution kernel with splitting function moments.

References
----------
- Vogelsang, W. (1998). Phys. Rev. D 57, 1886-1894
- Sha, C.M. & Ma, B. (2025). arXiv:2409.00221
"""
__docformat__ = "numpy"
from .constants import constants
import mpmath as mp
import numpy as np
from scipy.interpolate import interp1d as interp
from mpmath import invertlaplace, mpc, pi, zeta, psi, euler as euler_gamma

# Set the precision of mpmath to 15 decimal digits
mp.dps = 15

# Define commonly used constants
zeta2 = zeta(2)
zeta3 = zeta(3)



# Define the first few derivatives of the polygamma function
def psi0(s):
    """Digamma function ψ₀(s).

    Parameters
    ----------
    s : complex
        Complex argument.

    Returns
    -------
    complex
        Value of ψ₀(s).
    """
    return psi(0, s)

def psi_p(s):
    """Polygamma ψ′(s) (first derivative of digamma).

    Parameters
    ----------
    s : complex
        Complex argument.

    Returns
    -------
    complex
        Value of ψ′(s).
    """
    return psi(1, s)

def psi_pp(s):
    """Polygamma ψ″(s) (second derivative of digamma).

    Parameters
    ----------
    s : complex
        Complex argument.

    Returns
    -------
    complex
        Value of ψ″(s).
    """
    return psi(2, s)

# Define special functions which analytically continue the zeta function
# Eq. (28)
def S_1(n):
    """Harmonic sum S₁(n) analytically continued.

    Parameters
    ----------
    n : complex
        Mellin moment number.

    Returns
    -------
    complex
        Value of S₁(n).
    """
    return euler_gamma + psi0(n + 1)
# Eq. (29)
def S_2(n):
    """Harmonic sum S₂(n) analytically continued.

    Parameters
    ----------
    n : complex
        Mellin moment number.

    Returns
    -------
    complex
        Value of S₂(n).
    """
    return zeta2 - psi_p(n + 1)
# Eq. (30)
def S_3(n):
    """Harmonic sum S₃(n) analytically continued.

    Parameters
    ----------
    n : complex
        Mellin moment number.

    Returns
    -------
    complex
        Value of S₃(n).
    """
    return zeta3 + 0.5 * psi_pp(n + 1)

# Define eta ^ N as efficiently as possible, since the power function calls transcendental functions
def etaN(n, eta):
    """Compute η^n efficiently.

    Parameters
    ----------
    n : int or complex
        Exponent (Mellin moment index).
    eta : int
        Base, typically ±1 for plus/minus distributions.

    Returns
    -------
    complex
        η raised to the n-th power.
    """
    return 1 if eta == 1 else mp.power(eta, n)


def S_p1(n, f):
    """Compute first-order polarized harmonic sum S'_1.
    
    Implements Eq. (28) using the interpolation formula Eq. (31).
    
    Parameters
    ----------
    n : complex
        Mellin moment number.
    f : complex
        Factor η^N where η = ±1 for plus/minus distributions.
    
    Returns
    -------
    complex
        Polarized harmonic sum S'_1(N).
    """
    return 0.5 * (
        (1 + f) * S_1(n/2) + (1 - f) * S_1((n-1)/2))

def S_p2(n, f):
    """Compute second-order polarized harmonic sum S'_2.
    
    Implements Eq. (29) using the interpolation formula Eq. (31).
    
    Parameters
    ----------
    n : complex
        Mellin moment number.
    f : complex
        Factor η^N where η = ±1 for plus/minus distributions.
    
    Returns
    -------
    complex
        Polarized harmonic sum S'_2(N).
    """
    return 0.5 * (
        (1 + f) * S_2(n/2) + (1 - f) * S_2((n-1)/2))

def S_p3(n, f):
    """Compute third-order polarized harmonic sum S'_3.
    
    Implements Eq. (30) using the interpolation formula Eq. (31).
    
    Parameters
    ----------
    n : complex
        Mellin moment number.
    f : complex
        Factor η^N where η = ±1 for plus/minus distributions.
    
    Returns
    -------
    complex
        Polarized harmonic sum S'_3(N).
    """
    return 0.5 * (
        (1 + f) * S_3(n/2) + (1 - f) * S_3((n-1)/2))

# Define the part of Eq. (32) which depends on psi0
def G(n):
    """Auxiliary function G(n) = ψ₀((n+1)/2) − ψ₀(n/2).

    Parameters
    ----------
    n : complex
        Mellin moment number.

    Returns
    -------
    complex
        Value of G(n).
    """
    return psi0((n + 1) / 2) - psi0(n / 2)

def Stilde(n, f):
    """Compute the S-tilde harmonic sum function.
    
    Implements Eq. (32) which appears in the NLO splitting function moment.
    This function involves Riemann zeta function ζ(3), dilogarithm integral,
    and digamma function ψ_0.
    
    Parameters
    ----------
    n : complex
        Mellin moment number.
    f : complex
        Factor η^N where η = ±1 for plus/minus distributions.
    
    Returns
    -------
    complex
        S-tilde value at moment N.
    """
    temp = -5/8 * zeta3
    term = f
    term *= S_1(n) / n / n - zeta2/2 * G(n) + \
        mp.quad(lambda t: mp.power(t, n-1) * mp.polylog(2, t) / (1 + t), [0, 1])
    return temp + term

def LO_splitting_function_moment(n, CF):
    """Compute the leading-order splitting function Mellin moment.
    
    Implements Eq. (26) for the LO transversity splitting function moment.
    Uses harmonic sum S₁(n) defined via polygamma functions.
    
    Parameters
    ----------
    n : complex
        Mellin moment number.
    CF : float
        Color factor CF = (NC² - 1)/(2NC).
    
    Returns
    -------
    complex
        LO splitting function moment M[ΔT P_qq^(0)](n).
    """
    return CF * (1.5 - 2 * S_1(n))

def NLO_splitting_function_moment(n, eta, CF, NC, Tf):
    """Compute the next-to-leading-order splitting function Mellin moment.
    
    Implements Eq. (27) for the NLO transversity splitting function moment.
    Includes CF², CF×NC, and CF×Tf terms with harmonic sums.
    More complex than LO due to two-loop corrections.
    
    Parameters
    ----------
    n : complex
        Mellin moment number.
    eta : int
        Distribution type: 1 for plus, -1 for minus.
    CF : float
        Color factor.
    NC : int
        Number of colors.
    Tf : float
        Flavor factor TR × Nf.
    
    Returns
    -------
    complex
        NLO splitting function moment M[ΔT P_qq,η^(1)](n).
    """
    f = etaN(n, eta)
    return \
        CF * CF * (
            3 / 8
            + (1-eta) / (n * (n + 1))
            - 3 * S_2(n)
            - 4 * S_1(n) * (S_2(n) - S_p2(n, f))
            - 8 * Stilde(n, f)
            + S_p3(n, f)
        ) + \
        0.5 * CF * NC * (
            17 / 12
            - (1 - eta) / (n * (n + 1))
            - 134 / 9 * S_1(n)
            + 22 / 3 * S_2(n)
            + 4 * S_1(n) * (2 * S_2(n) - S_p2(n, f))
            + 8 * Stilde(n, f)
            - S_p3(n, f)
        ) + \
        2 / 3 * CF * Tf * (
            -1 / 4
            + 10 / 3 * S_1(n)
            - 2 * S_2(n)
        )

def alpha_S(Q2, order, beta0, beta1, l_QCD):
    """Compute the strong coupling constant using the analytical approximation.
    
    Implements Eq. (4) from the paper for α_s(Q²).
    The NLO approximation includes the β₁ correction term.
    
    Parameters
    ----------
    Q2 : float
        Energy scale squared (GeV²).
    order : int
        Perturbative order: 1 for LO, 2 for NLO.
    beta0 : float
        Leading QCD beta function coefficient.
    beta1 : float
        Next-to-leading QCD beta function coefficient.
    l_QCD : float
        QCD scale parameter Λ (GeV).
    
    Returns
    -------
    float
        Strong coupling constant α_s(Q²).
    """
    ln_Q2_L_QCD = mp.log(Q2) - 2 * mp.log(l_QCD)
    ln_ln_Q2_L_QCD = mp.log(ln_Q2_L_QCD)
    alpha_S = 4 * pi / beta0 / ln_Q2_L_QCD
    if order == 2:
        alpha_S -= 4 * pi * beta1 / mp.power(beta0, 3) * ln_ln_Q2_L_QCD / ln_Q2_L_QCD / ln_Q2_L_QCD
    return alpha_S

def alpha_S_num(Q2, order, Q0_2_a, a0, beta0, beta1):
    """Compute the strong coupling constant via numerical ODE evolution.
    
    Solves the QCD beta function differential equation numerically for α_s(Q²).
    More accurate than the analytical approximation, especially at low scales.
    Uses mpmath's odefun for high-precision ODE integration.
    
    Parameters
    ----------
    Q2 : float
        Energy scale squared (GeV²).
    order : int
        Perturbative order: 1 for LO, 2 for NLO.
    Q0_2_a : float
        Reference energy scale squared where α_s is known (GeV²).
    a0 : float
        Reference coupling α_s(Q0_2_a)/(4π).
    beta0 : float
        Leading QCD beta function coefficient.
    beta1 : float
        Next-to-leading QCD beta function coefficient.
    
    Returns
    -------
    float
        Strong coupling constant α_s(Q²).
    """
    if order == 2:
        ode = lambda x, a: -beta0 * a * a - beta1 * a * a * a
    else:
        ode = lambda x, a: -beta0 * a * a
    if Q2 < Q0_2_a:
        ode_fixed = lambda x, a: -ode(x, a)
        f = mp.odefun(ode_fixed, -mp.log(Q0_2_a), a0)
        return f(-np.log(Q2)) * 4 * pi
    else:
        f = mp.odefun(ode, mp.log(Q0_2_a), a0)
        return f(np.log(Q2)) * 4 * pi

def mellin(f, s):
    """Compute the Mellin transform of a function.
    
    Implements Eq. (20): M[f](s) = ∫₀¹ t^(s-1) f(t) dt.
    Uses mpmath.quad for high-precision integration.
    
    Parameters
    ----------
    f : callable
        Function to transform, defined on [0, 1].
    s : complex
        Mellin moment number.
    
    Returns
    -------
    complex
        Mellin transform M[f](s).
    """
    return mp.quad(lambda t: mp.power(t, s-1) * f(t), [0, 1])

def inv_mellin(f, x, degree=5, verbose=True):
    """Compute the inverse Mellin transform using Cohen contour method.
    
    Implements Eq. (36) for reconstructing the PDF from its Mellin moments.
    Uses mpmath's invertlaplace with the 'cohen' method for optimal
    convergence. Higher degree values increase accuracy but slow computation.
    
    Parameters
    ----------
    f : callable
        Function in Mellin space, f(s).
    x : float
        Point in x-space where to evaluate the inverse transform.
    degree : int
        Number of terms in Cohen's convergence acceleration (default: 5).
    verbose : bool
        Print intermediate results if True (default: True).
    
    Returns
    -------
    float
        Inverse Mellin transform value at x.
    
    References
    ----------
    Cohen et al. (2000), Experiment. Math. 9(1): 3-12
    """
    res = invertlaplace(f, -mp.log(x), method='cohen', degree=degree)
    if verbose:
        print(x, x*res)
    return res

def evolveMoment(n, pdf_m, alpha_S_Q0_2, alpha_S_Q2, beta0, beta1, eta, CF, NC, Tf):
    """Evolve a single Mellin moment from initial to final energy scale.
    
    Implements Eq. (24) from the paper (Vogelsang's formula).
    Combines LO and NLO splitting function moments with running coupling.
    
    Parameters
    ----------
    n : complex
        Mellin moment number.
    pdf_m : complex
        Mellin moment of input PDF at initial scale.
    alpha_S_Q0_2 : float
        Strong coupling at initial scale Q₀².
    alpha_S_Q2 : float
        Strong coupling at final scale Q².
    beta0 : float
        Leading QCD beta function coefficient.
    beta1 : float
        Next-to-leading QCD beta function coefficient.
    eta : int
        Distribution type: 1 for plus, -1 for minus.
    CF : float
        Color factor.
    NC : int
        Number of colors.
    Tf : float
        Flavor factor.
    
    Returns
    -------
    complex
        Evolved Mellin moment at scale Q².
    """
    total = 1
    total += (alpha_S_Q0_2 - alpha_S_Q2) / pi / beta0 * (NLO_splitting_function_moment(n, eta, CF, NC, Tf) - beta1 / 2 / beta0 * LO_splitting_function_moment(n, CF))
    total *= mp.power(alpha_S_Q2 / alpha_S_Q0_2, -2 / beta0 * LO_splitting_function_moment(n, CF)) * pdf_m
    return total

def evolve(
    pdf: np.ndarray,
    Q0_2: float = 0.16,
    Q2: float = 5.0,
    l_QCD: float = 0.25,
    n_f: int = 5,
    CG: float = 3,
    morp: str = 'minus',
    order: int = 2,
    n_x: int = 200,
    verbose: bool = False,
    Q0_2_a: float = 91.1876**2,
    a0: float = 0.118 / 4 / pi,
    alpha_num: bool = True,
    degree: int = 5,
) -> np.ndarray:
    """Evolve transversity PDF using the Mellin moment method (Vogelsang).
    
    This is the main function for PDF evolution using Mellin transforms.
    Faster and less discretization-dependent than the direct integration method.
    
    This method:
    1. Computes Mellin moments of input PDF
    2. Evolves moments using splitting function moments (Eq. 24)
    3. Reconstructs PDF via inverse Mellin transform (Cohen method)
    
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
    morp : str, optional
        Distribution type (default: 'minus'). Options are 'plus'
        (ΔT q⁺ = ΔT u + ΔT d) or 'minus' (ΔT q⁻ = ΔT u - ΔT d).
    order : int, optional
        Perturbative order (default: 2). Use 1 for LO or 2 for NLO.
    n_x : int, optional
        Number of x grid points minus 1 for output (default: 200).
    verbose : bool, optional
        Print (x, x*pdf(x)) during evolution if True (default: False).
    Q0_2_a : float, optional
        Reference scale Q₀² where αs is known, in GeV² (default: 91.1876²).
        Only used if alpha_num=True.
    a0 : float, optional
        Reference coupling αs(Q0_2_a)/(4π) (default: 0.118/(4π)).
        Only used if alpha_num=True.
    alpha_num : bool, optional
        Use numerical ODE evolution for αs if True (default: True).
        If False, uses analytical approximation.
    degree : int, optional
        Convergence acceleration degree for inverse Mellin (default: 5).
        Higher values increase accuracy but slow computation.
    
    Returns
    -------
    ndarray
        Evolved PDF as 2D array [[x, x*f_evolved(x)], ...]. Shape: (n_x+2, 2)
        due to padding at boundaries.
    
    Notes
    -----
    Advantages over direct integration (t_evolution.evolve):
    - Typically 10-100x faster
    - Less sensitive to discretization
    - Better for smooth PDFs
    
    Disadvantages:
    - Less direct control over integration
    - May have issues with very peaked PDFs
    
    Examples
    --------
    >>> import numpy as np
    >>> from tparton.m_evolution import evolve
    >>> x = np.linspace(0, 1, 100)
    >>> pdf_in = x * (1-x)**3  # x*f(x) format
    >>> pdf_out = evolve(pdf_in, Q0_2=4.0, Q2=100.0, order=2)
    >>> x_out, xf_out = pdf_out[:, 0], pdf_out[:, 1]
    
    See Also
    --------
    t_evolution.evolve : Direct integration method (Hirai)
    evolveMoment : Evolves a single Mellin moment
    inv_mellin : Inverse Mellin transform (Cohen method)
    
    References
    ----------
    - Vogelsang, W. (1998). Phys. Rev. D 57, 1886-1894
    - Sha, C.M. & Ma, B. (2025). arXiv:2409.00221
    """

    if pdf.shape[-1] == 1:
        # If only the x*pdf(x) values are supplied, assume a linear spacing from 0 to 1
        xs = np.linspace(0, 1, len(pdf))
    else:
        # Otherwise split the input array
        xs, pdf = pdf[:, 0], pdf[:, 1]
    
    # Divide x*pdf(x) by x. 
    # In the Hirai method, the evolution of x*pdf(x) and pdf(x) are numerically identical and do not require this extra step.
    pdf = pdf / (xs + 1e-100)
    # We assume that pdf(0) = 0
    pdf[0] = 0

    # Interpolate the resulting (x, pdf(x)) pairs as a function
    pdf_fun = interp(xs, pdf, fill_value=0, assume_sorted=True)
    
    # Convert the pdf into one compatible with mpmath's internal floating point representation
    pdf = lambda x: mp.mpf(pdf_fun(float(x)).item())

    # The type of distribution determines eta in Eq. (31)
    eta = 1 if morp == 'plus' else -1

    # Calculate the color constants
    NC, CF, Tf, beta0, beta1 = constants(CG, n_f)

    if order == 1:
        # If the desired order of accuracy is LO, we simply set beta1 to 0, which reproduces the relevant LO equations
        beta1 = 0
        # For the sake of efficiency, we also redefine the NLO splitting function moment to be the zero function
        NLO_splitting_function_moment = lambda n, eta, CF, NC, Tf: 0
    
    if alpha_num:
        # Use the numerically evolved alpha_S
        alpha_S_Q0_2 = alpha_S_num(Q0_2, order, Q0_2_a, a0, beta0, beta1)
        alpha_S_Q2 = alpha_S_num(Q2, order, Q0_2_a, a0, beta0, beta1)
    else:
        # Use the approximate analytical expression for alpha_S in Eq. (4)
        alpha_S_Q0_2 = alpha_S(Q0_2, order, beta0, beta1, l_QCD)
        alpha_S_Q2 = alpha_S(Q2, order, beta0, beta1, l_QCD)

    # Choose the values of x at which the evolved pdf(x) will be evaluated
    if n_x > 0:
        xs = np.linspace(0, 1, n_x+2)
    # In all cases, we assume that xs[0] = 0 and xs[1] = 1, pdf(0) = pdf(1) = 0, so no evolution is necessary at these points.
    # Even if pdf(0) != 0, this slight change will not significantly affect the final numerical result.
    xs = xs[1:-1]

    # A function representing the Mellin transform of pdf(x), Eq. (20)
    pdf_m = lambda s: mellin(pdf, s)
    # A function representing the resulting evolved moments, Eq. (24)
    pdf_evolved_m = lambda s: mpc(evolveMoment(s, pdf_m(s), alpha_S_Q0_2, alpha_S_Q2, beta0, beta1, eta, CF, NC, Tf))
    # Perform Mellin inversion on the evolved moments, Eq. (36)
    pdf_evolved = np.array([inv_mellin(pdf_evolved_m, x, degree=degree, verbose=verbose).__complex__().real for x in xs])

    # Reinstate the endpoints x = 0 and x = 1
    xs = np.pad(xs, 1)
    xs[-1] = 1

    # Pad the evolved pdf so that pdf(0) = pdf(1) = 0
    pdf_evolved = np.pad(pdf_evolved, 1)
    # Organize the (x, x*pdf_evolved(x)) pairs into an array
    pdf_evolved = np.stack((xs, np.array(xs) * np.array(pdf_evolved)), axis=1)
    print('Done!')
    return pdf_evolved

def main():
    """Command-line interface for Mellin moment evolution method.
    
    Parses command-line arguments and runs PDF evolution using Vogelsang's method.
    This function is called when running `python -m tparton m`.
    
    See Also:
        evolve:
    """
    import argparse
    parser = argparse.ArgumentParser(description='Evolution of the nonsinglet transversity PDF, using Vogelsang\'s moment method.')
    parser.add_argument('type',action='store',type=str,help='The method you chose')
    parser.add_argument('input', action='store', type=str,
                    help='The CSV file containing (x,x*PDF(x)) pairs on each line. If only a single number on each line, we assume a linear spacing for x between 0 and 1 inclusive')
    parser.add_argument('Q0sq', action='store', type=float, help='The starting energy scale in units of GeV^2')
    parser.add_argument('Qsq', action='store', type=float, help='The ending energy scale in units of GeV^2')
    parser.add_argument('--morp', nargs='?', action='store', type=str, default='plus', help='The plus vs minus type PDF (default is \'plus\')')
    parser.add_argument('-o', action='store', nargs='?', type=str, default='out.dat', help='Output file for the PDF, stored as (x,x*PDF(x)) pairs.')
    parser.add_argument('-l', metavar='l_QCD', nargs='?', action='store', type=float, default=0.25, help='The QCD scale parameter (default 0.25 GeV^2). Only used when --alpha_num is False.')
    parser.add_argument('--n_f', metavar='n_f', nargs='?', action='store', type=int, default=5, help='The number of flavors (default 5)')
    parser.add_argument('--nc', metavar='n_c', nargs='?', action='store', type=int, default=3, help='The number of colors (default 3)')
    parser.add_argument('--order', metavar='order', nargs='?', action='store', type=int, default=2, help='1: leading order, 2: NLO DGLAP (default 2)')
    parser.add_argument('--nx', metavar='n_x', nargs='?', action='store', type=int, default=-1, help='The number of x values to sample the evolved PDF (default -1). If left at -1, will sample at input xs.')
    parser.add_argument('--alpha_num', metavar='alpha_num', nargs='?', action='store', type=bool, default=True, help='Set to use the numerical solution for the strong coupling constant, numerically evolved at LO or NLO depending on the --order parameter.')
    parser.add_argument('--Q0sqalpha', metavar='Q0sqalpha', nargs='?', action='store', type=float, default=91.1876**2, help='The reference energy squared at which the strong coupling constant is known. Default is the squared Z boson mass. Use in conjunction with --a0. Only used when --alpha_num is True.')
    parser.add_argument('--a0', metavar='a0', nargs='?', action='store', type=float, default=0.118 / 4 / np.pi, help='The reference value of the strong coupling constant a = alpha / (4 pi) at the corresponding reference energy --Q0sqalpha. Default is 0.118 / (4 pi), at energy Q0sqalpha = Z boson mass squared. Only used when --alpha_num is True.')
    parser.add_argument('--delim', nargs='?', action='store', type=str, default=' ', help='Delimiter for the output (default \' \'). If given without an argument, then the delimiter is whitespace (i.e. Mathematica output.)')
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
    n_f = args.n_f
    nc = args.nc
    order = args.order
    nx = args.nx
    alpha_num = args.alpha_num
    Q0sqalpha = args.Q0sqalpha
    a0 = args.a0
    verbose = args.v

    res = evolve(pdf,
        Q0_2=Q0sq,
        Q2=Qsq,
        l_QCD=l,
        n_f=n_f,
        CG=nc,
        morp=morp,
        order=order,
        n_x=nx,
        verbose=verbose,
        alpha_num=alpha_num,
        Q0_2_a=Q0sqalpha,
        a0=a0
    )

    np.savetxt(args.o, res.T, delimiter=args.delim)

if __name__ == '__main__':
    main()