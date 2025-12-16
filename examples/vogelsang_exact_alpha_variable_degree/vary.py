from tparton.m_evolution import evolve
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

def A(a, b, g, rho):
    return (1 + g * a / (a + b + 1)) \
        * gamma(a) * gamma(b+1) / gamma(a + b + 1) \
        + rho * gamma(a + 0.5) * gamma(b + 1) / gamma(a + b + 1.5)

def pdf_u(x, eta_u=0.918, a_u=0.512, b_u=3.96, gamma_u=11.65, rho_u=-4.60):
    return eta_u / A(a_u, b_u, gamma_u, rho_u) * np.power(x, a_u) * np.power(1-x, b_u) * (1 + gamma_u * x + rho_u * np.sqrt(x))

def pdf_d(x, eta_d=-0.339, a_d=0.780, b_d=4.96, gamma_d=7.81, rho_d=-3.48):
    return pdf_u(x, eta_d, a_d, b_d, gamma_d, rho_d)

def vary(degree, N):
    print(degree)
    n = 300
    x = np.power(10, np.linspace(np.log10(1/N), 0, N))
    x = np.concatenate(([0], x))
    u = np.stack((x, pdf_u(x))).T
    d = np.stack((x, pdf_d(x))).T
    u_evolved = evolve(u, Q0_2=4, Q2=200, n_x=n, l_QCD=0.231, n_f=4, morp='minus', verbose=True, degree=degree)
    d_evolved = evolve(d, Q0_2=4, Q2=200, n_x=n, l_QCD=0.231, n_f=4, morp='minus', verbose=True, degree=degree)
    np.savez(f'vogelsang-exact_alpha_deg_{degree}_N_{N}.npz', x=x, upd=u[:,1]+d[:,1], u_evolved=u_evolved, d_evolved=d_evolved)
    return np.load(f'vogelsang-exact_alpha_deg_{degree}_N_{N}.npz')