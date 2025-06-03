# Copyright Congzhou M Sha 2024
from .constants import constants
import numpy as np
from scipy.integrate import simpson
from numpy._core.multiarray import interp
from scipy.integrate import odeint
from scipy.special import spence

pi = np.pi

def alp2pi(t, lnlam, order, beta0, beta1):
    '''The approximation expression of alpha / 2 pi, Eq. (4)'''
    dlnq2 = t - lnlam
    alpha = 4 * pi / beta0 / dlnq2
    alpha_factor = 1 if order == 1 else (1 - beta1 * np.log(dlnq2) / beta0**2 / dlnq2)
    alpha_factor /= (2 * pi)
    return alpha_factor * alpha

def splitting(z, CF, order, sign, CG, Tf):
    '''Gives the LO and NLO splitting functions evaluated at x = z as a function of theory constants.
    z is assumed to be a 1D array so that this function is efficiently vectorized.'''

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
        # For NLO terms in the splitting function, evaluate Eq. (11) at x = z
        # log z
        lnz = np.log(z)
        # log (1 - z)
        lno = np.log(omz+1e-100)
        # First line of Eq. (12)
        dP0 = 2 * z / (omz+1e-100)
        # The terms excluding the plus and delta function terms in the first line of Eq. (12)
        pp1 = omz - (3 / 2 + 2 * lno) * lnz * dP0
        # The terms excluding the plus and delta function terms in the second line of Eq. (12)
        pp2 = -omz + (67/9 + 11/3 * lnz + lnz**2 - pi**2 / 3) * dP0
        # The terms excluding the plus and delta function terms in the third line of Eq. (12)
        pp3 = (-lnz - 5/3) * dP0

        # Summing the non-delta function terms in Eq. (12)
        dpqq = CF * CF * pp1 + CF * CG * 0.5 * pp2 + 2 / 3 * CF * Tf * pp3

        # Implementation of Eq. (13)
        pp4 = -omz + 2 * s2 * 2 * -z / (1 + z)
        dpqqb = CF * (CF-CG / 2) * pp4
        # Adding the additional contribution from Eq. (13) to Eq. (12)
        p1 = dpqq + sign * dpqqb
        # Avoid the possible singularity at x = 0
        p1[0] = 0
    else:
        # At LO, the NLO terms are 0
        p1 = 0
    
    #######################################################################################################
    # Plus function, f(x) g(x)_+ in Eq. (10), contributions to the splitting function
    #######################################################################################################
    # Enforcing the plus prescriptions for the relevant terms in Eq. (12)
    # The plus prescription in line 0 of Eq. (12) results in 0, so it is ignored
    # Line 2 plus prescription in Eq. (12)
    p2plus = -(67/9-pi**2/3) * 2 / (omz+1e-100)
    # Line 3 plus prescription in Eq. (12)
    p3plus = 5 / 3 * 2 / (omz+1e-100)
    # The plus function contributions from all three lines in Eq. (12)
    p1pf = CF * CG / 2 * p2plus + 2 / 3 * CF * Tf * p3plus

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
        # Hard code zta = zeta(3)
        zta = 1.2020569031595943
        # The delta function contributions from Eq. (12), at x = 1
        del1 = CF * CF * (3 / 8 - pi**2 / 2 + 6 * zta) + \
            CF * CG / 2 * (17 / 12 + 11 * pi**2 /9 - 6 * zta) - \
            2 / 3 * CF * Tf * (1 / 4 + pi**2 / 3)
        # The plus prescription terms from Eq. (12)
        # Again, the plus prescription in line 0 of Eq. (12) results in 0, so it is ignored
        p2pl = (67 / 9 - pi**2/3) * 2
        p3pl = -5 / 3 * 2
        # The sum of the f(1)g(x) plus parts
        plus1 = CF * CG / 2 * p2pl + 2 / 3 * CF * Tf * p3pl
    else:
        # At LO, the NLO terms are 0
        plus1 = 0
        del1 = 0

    return p0, p1, p0pf, p1pf, plus0, del0, plus1, del1

# Define the integration step required here
def integrate(pdf, i, z, alp, order, CF, sign, CG, Tf, xs):
    '''Performs the convolution of pdf(x) with the splitting function at x = z.
    z is assumed to be a 1D array so that this function is efficiently vectorized.'''

    # Handle the base case of an empty array
    if len(z) == 0:
        return 0
    
    # Evaluate the splitting function at the points z
    p0, p1, p0pf, p1pf, plus0, del0, plus1, del1 = splitting(z, CF, order, sign, CG, Tf)

    # Implement Eq. (19), instead of Eq. (7) for the convolution
    func = ((p0 + (alp * p1 if order == 2 else 0)) * interp(xs[i] / z, xs, pdf)) + \
        (p0pf + (alp * p1pf if order == 2 else 0)) * pdf[i]

    # When handling the plus prescription, there is a common factor of ln(1-x) when integrating Eq. (10)
    lno = np.log(1 - xs[i])
    estimate = simpson(func, x=z) + (plus0 * lno + del0) * pdf[i]
    if order == 2:
        estimate += alp * (plus1 * lno + del1) * pdf[i]

    return estimate

def evolve(
    pdf,
    Q0_2=0.16,
    Q2=5.0,
    l_QCD=0.25,
    n_f=5,
    CG=3,
    n_t=100,
    n_z=500,
    morp='plus',
    order=2,
    logScale=False,
    verbose=False,
    Q0_2_a=91.1876 ** 2,
    a0=0.118 / 4 / np.pi,
    alpha_num=True
):
    '''
    Evolve the transversity PDF
    ******************************
    Parameters:
        pdf: array-like
            the input first moment (assumed to be at x
            evenly distributed on [0, 1] inclusive)

        Q0_2: float
            initial energy scale squared

        Q2: float
            final evolved energy scale squared

        l_QCD: float
            QCD energy scale, if using the approximate expression for alpha in Eq. (4)

        n_f: int
            number of flavors

        CG: float
            number of colors

        n_t: int
            number of Euler time steps by which to integrate Eq. (1)

        n_z: int
            the number of z steps to approximate in convolutions given by Eq. (3)

        morp: 'plus' or 'minus'
            type of pdf (plus or minus type)

        order: int
            1: first-order (LO)
            2: second-order (NLO)

        logScale: bool
            the integration points z are log scaled between 0 and 1 if True
            otherwise the integration points are linearly scaled

        verbose: bool
            prints the number of time steps so far if True
        
        Q0_2_a: float
            the reference energy squared at which the strong coupling constant a0 is known
            only used if alpha_num is True
            default is the Z boson mass squared
        
        a0: float
            the reference strong coupling constant at the energy scale Q0_2_a
            only used if alpha_num is True
            default is a0 = 0.118 / (4 pi) at the Z boson mass squared
        
        alpha_num: bool
            uses the numerically evolved coupling constant rather than Eq. (4) if True
        
    '''
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
    
    if order == 2:
        ode = lambda x, a: -beta0 * a * a - beta1 * a * a * a
    else:
        ode = lambda x, a: -beta0 * a * a

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
        inc = np.array([integrate(res, index, \
            np.power(10, np.linspace(np.log10(xs[index]), 0, n_z + 1)) if logScale else np.linspace(xs[index], 1, n_z + 1), \
                alp, order, CF, sign, CG, Tf, xs) for index in range(1, len(xs)-1)])
        # Ensure that x*pdf(x) = 0 at x = 0 and x = 1
        inc = np.pad(inc, 1)
        res += dt * inc * alp
    return np.stack((xs, res))


def main():
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
