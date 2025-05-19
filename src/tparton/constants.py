def constants(CG, n_f):
    '''Define the commonly used constants in terms of the number of colors CG and number of flavors n_f.
    These constants are defined after Eq. (4) in the manuscript.'''
    NC = CG
    CF = (NC * NC - 1) / NC / 2
    TR = 1/2
    Tf = TR * n_f
    beta0 = 11 / 3 * CG - 4 / 3 * TR * n_f
    beta1 = 34 / 3 * CG ** 2 - 10 / 3 * CG * n_f - 2 * CF * n_f
    return NC, CF, Tf, beta0, beta1