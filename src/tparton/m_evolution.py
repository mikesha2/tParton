# Copyright Congzhou M Sha 2024
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
psi0 = lambda s: psi(0,s)
psi_p = lambda s: psi(1,s)
psi_pp = lambda s: psi(2,s)

# Define special functions which analytically continue the zeta function
# Eq. (28)
S_1 = lambda n: euler_gamma + psi0(n+1)
# Eq. (29)
S_2 = lambda n: zeta2 - psi_p(n+1)
# Eq. (30)
S_3 = lambda n: zeta3 + 0.5 * psi_pp(n+1)

# Define eta ^ N as efficiently as possible, since the power function calls transcendental functions
etaN = lambda n, eta: 1 if eta == 1 else mp.power(eta, n)


def S_p1(n, f):
    '''Define the derivative of the harmonic sum in Eq. (28) using Eq. (31).
    n is the moment and f = eta ^ N.'''
    return 0.5 * (
        (1 + f) * S_1(n/2) + (1 - f) * S_1((n-1)/2))

def S_p2(n, f):
    '''Define the derivative of the harmonic sum in Eq. (29) using Eq. (31).
    n is the moment and f = eta ^ N.'''
    return 0.5 * (
        (1 + f) * S_2(n/2) + (1 - f) * S_2((n-1)/2))

def S_p3(n, f):
    '''Define the derivative of the harmonic sum in Eq. (30) using Eq. (31).
    n is the moment and f = eta ^ N.'''
    return 0.5 * (
        (1 + f) * S_3(n/2) + (1 - f) * S_3((n-1)/2))

# Define the part of Eq. (32) which depends on psi0
G = lambda n: psi0((n+1)/2) - psi0(n/2)

def Stilde(n, f):
    '''Define S-tilde according to Eq. (32).
    n is the moment and f = eta ^ N.'''
    temp = -5/8 * zeta3
    term = f
    term *= S_1(n) / n / n - zeta2/2 * G(n) + \
        mp.quad(lambda t: mp.power(t, n-1) * mp.polylog(2, t) / (1 + t), [0, 1])
    return temp + term

def LO_splitting_function_moment(n, CF):
    '''The Mellin transform of the LO part of the splitting function. Corresponds to MDTP_qq_LO in Eq. (26).
    n is the moment and CF is the constant given previously.'''
    return CF * (1.5 - 2 * S_1(n))

def NLO_splitting_function_moment(n, eta, CF, NC, Tf):
    '''The Mellin transform of the NLO part of the splitting function. Corresponds to MDTP_qq_NLO in Eq. (27).
    n is the moment while eta, CF, NC, and Tf are the constants given previously.'''
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
    '''The approximate alpha_S given in Eq. (4), as a function of the energy scale Q^2, the order of approximation, and theory constants.'''
    ln_Q2_L_QCD = mp.log(Q2) - 2 * mp.log(l_QCD)
    ln_ln_Q2_L_QCD = mp.log(ln_Q2_L_QCD)
    alpha_S = 4 * pi / beta0 / ln_Q2_L_QCD
    if order == 2:
        alpha_S -= 4 * pi * beta1 / mp.power(beta0, 3) * ln_ln_Q2_L_QCD / ln_Q2_L_QCD / ln_Q2_L_QCD
    return alpha_S

def alpha_S_num(Q2, order, Q0_2_a, a0, beta0, beta1):
    '''The numerically evolved alpha_S, as a function of the energy scale squared Q^2, the order of approximation, and theory constants.'''
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
    '''The definition of Mellin transform of the function f(t) evaluated at s, Eq. (20).'''
    return mp.quad(lambda t: mp.power(t, s-1) * f(t), [0, 1])

def inv_mellin(f, x, degree=5, verbose=True):
    '''Wrap the inverse Mellin transform of a function f evaluated at x, Eq. (36).'''
    res = invertlaplace(f, -mp.log(x), method='cohen', degree=degree)
    if verbose:
        print(x, x*res)
    return res

def evolveMoment(n, pdf_m, alpha_S_Q0_2, alpha_S_Q2, beta0, beta1, eta, CF, NC, Tf):
    '''Implement Eq. (24) which evolves the moments from energy Q0^2 to Q^2 and evaluates the resulting function at n.'''
    total = 1
    total += (alpha_S_Q0_2 - alpha_S_Q2) / pi / beta0 * (NLO_splitting_function_moment(n, eta, CF, NC, Tf) - beta1 / 2 / beta0 * LO_splitting_function_moment(n, CF))
    total *= mp.power(alpha_S_Q2 / alpha_S_Q0_2, -2 / beta0 * LO_splitting_function_moment(n, CF)) * pdf_m
    return total

def evolve(
    pdf,
    Q0_2=0.16,
    Q2=5.0,
    l_QCD=0.25,
    n_f=5,
    CG=3,
    morp='minus',
    order=2,
    n_x=200,
    verbose=False,
    Q0_2_a=91.1876**2,
    a0=0.118 / 4 / pi,
    alpha_num=True,
    degree=5,
):
    '''
    Evolve the transversity PDF
    ******************************
    Parameters:
        pdf: 1D or 2D array-like
            the input first moment. If 1D, assumed to be at x
            evenly distributed on [0, 1] inclusive

        Q0_2: float
            initial energy scale squared

        Q2: float
            final evolved energy scale squared

        l_QCD: float
            QCD energy scale, if using the approximate expression for alpha in Eq. (4)

        n_f: int
            number of flavors of quarks

        CG: float
            number of colors
        
        morp: 'plus' or 'minus'
            type of pdf (plus or minus type)

        order: int
            1: first-order (LO)
            2: second-order (NLO)
        
        n_x: int
            the number of points (minus one) between 0 and 1 inclusive at which to evaluate the evolved PDF

        verbose: bool
            prints (x, x * pdf_evolved(x)) during evolution if True
        
        Q0_2_a: float
            the reference energy scale Q^2 at which the strong coupling constant a0 = alpha_S / (4 pi) is known
            only used if alpha_num is True
            default is the Z boson mass squared
        
        a0: float
            the reference strong coupling constant at the energy scale Q0_2_a
            only used if alpha_num is True
            default is a0 = 0.118 / (4 pi) at the Z boson mass squared
        
        alpha_num: bool
            uses the numerically evolved coupling constant rather than Eq. (4) if True
        
        degree: int
            the number of terms to which the inverse Laplace/Mellin transform is approximated as an alternating series,
            as described by Cohen et al. in "Convergence acceleration of alternating series," Experiment. Math. 9(1): 3-12 (2000).
    '''

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
    pdf_evolved = np.stack((xs, np.array(xs) * np.array(pdf_evolved)))
    print('Done!')
    return pdf_evolved

def main():
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