"""Test that evolution methods produce consistent results with reference data.

These tests ensure that code changes (especially documentation changes) have not
affected the numerical results of the evolution methods.
"""
import numpy as np
import pytest
from scipy.special import gamma


def A(a, b, g, rho):
    """Helper function for PDF parameterization (Gehrmann)."""
    return (1 + g * a / (a + b + 1)) \
        * gamma(a) * gamma(b+1) / gamma(a + b + 1) \
        + rho * gamma(a + 0.5) * gamma(b + 1) / gamma(a + b + 1.5)


def pdf_u(x, eta_u=0.918, a_u=0.512, b_u=3.96, gamma_u=11.65, rho_u=-4.60):
    """u quark transversity PDF at initial scale (Gehrmann parameterization)."""
    return eta_u / A(a_u, b_u, gamma_u, rho_u) * np.power(x, a_u) * \
        np.power(1-x, b_u) * (1 + gamma_u * x + rho_u * np.sqrt(x))


def pdf_d(x, eta_d=-0.339, a_d=0.780, b_d=4.96, gamma_d=7.81, rho_d=-3.48):
    """d quark transversity PDF at initial scale (Gehrmann parameterization)."""
    return pdf_u(x, eta_d, a_d, b_d, gamma_d, rho_d)


@pytest.fixture
def input_pdf():
    """Create input PDF on logarithmic grid."""
    n = 100
    x = np.power(10, np.linspace(np.log10(1/100), 0, n))
    x = np.concatenate(([0], x))
    
    # Create u-d difference as input
    y = pdf_u(x) - pdf_d(x)
    diff = np.stack((x, y)).T
    return diff


@pytest.fixture
def reference_data_hirai():
    """Load reference data for Hirai method."""
    data = np.load('examples/hirai_approx.npz')
    return data


@pytest.fixture
def reference_data_vogelsang():
    """Load reference data for Vogelsang method."""
    data = np.load('examples/vogelsang_approx.npz')
    return data


def test_hirai_method_consistency(input_pdf, reference_data_hirai):
    """Test that Hirai (direct integration) method produces expected results.
    
    This test evolves a transversity PDF using the direct integration method
    and verifies the code runs successfully. Note: Exact numerical comparison
    with reference data is not performed because we use a coarser grid (n=100)
    for faster testing, while reference data was generated with n=3000.
    """
    from tparton.t_evolution import evolve
    
    # Evolution parameters matching the reference data generation
    result = evolve(
        input_pdf,
        Q0_2=4.0,
        Q2=200.0,
        l_QCD=0.231,
        n_f=4,
        n_t=100,
        morp='plus',
        logScale=True,
        alpha_num=False
    )
    
    # Extract x and evolved PDF values
    x_result = result[0]
    pdf_result = result[1]
    
    # Verify basic properties
    assert len(x_result) == 101, "Should have 101 x values"
    assert x_result[0] == 0.0, "First x value should be 0"
    assert x_result[-1] == 1.0, "Last x value should be 1"
    assert pdf_result[0] == 0.0, "PDF should be 0 at x=0"
    assert pdf_result[-1] == 0.0, "PDF should be 0 at x=1"
    assert np.all(pdf_result >= 0) or np.all(pdf_result <= 0), "PDF should have consistent sign"


def test_vogelsang_method_consistency(input_pdf, reference_data_vogelsang):
    """Test that Vogelsang (Mellin moment) method produces expected results.
    
    This test evolves a transversity PDF using the Mellin moment method
    and verifies the code runs successfully. Note: Exact numerical comparison
    with reference data is not performed because we use a coarser grid (n_x=300)
    for faster testing, while reference data was generated with n_x=3000.
    """
    from tparton.m_evolution import evolve
    
    # Evolution parameters matching the reference data generation
    result = evolve(
        input_pdf,  # Vogelsang method accepts 2D array
        Q0_2=4.0,
        Q2=200.0,
        l_QCD=0.231,
        n_f=4,
        morp='minus',
        n_x=300,
        alpha_num=False
    )
    
    # Extract x and evolved PDF values
    x_result = result[0]
    pdf_result = result[1]
    
    # Verify basic properties
    assert len(x_result) == 302, "Should have 302 x values (n_x+2 due to padding)"
    assert x_result[0] == 0.0, "First x value should be 0"
    assert x_result[-1] == 1.0, "Last x value should be 1"
    assert pdf_result[0] == 0.0, "PDF should be 0 at x=0"
    # Note: Vogelsang method may not enforce PDF=0 at x=1 exactly
    assert np.all(pdf_result >= 0) or np.all(pdf_result <= 0), "PDF should have consistent sign"


def test_both_methods_give_similar_results(input_pdf):
    """Test that both methods give similar results for the same evolution.
    
    While the methods use different numerical approaches, they should
    give similar results for the same physical evolution.
    """
    from tparton.t_evolution import evolve as t_evolve
    from tparton.m_evolution import evolve as m_evolve
    
    # Common parameters
    Q0_2, Q2 = 4.0, 200.0
    l_QCD, n_f = 0.231, 4
    
    # Evolve with Hirai method
    result_hirai = t_evolve(
        input_pdf,
        Q0_2=Q0_2,
        Q2=Q2,
        l_QCD=l_QCD,
        n_f=n_f,
        n_t=100,
        morp='minus',
        logScale=True,
        alpha_num=False
    )
    
    # Evolve with Vogelsang method
    result_vogelsang = m_evolve(
        input_pdf,  # Vogelsang accepts 2D array
        Q0_2=Q0_2,
        Q2=Q2,
        l_QCD=l_QCD,
        n_f=n_f,
        morp='minus',
        n_x=300,
        alpha_num=False
    )
    
    # Compare results (they should agree within a few percent due to different numerics)
    # Interpolate Vogelsang result onto Hirai grid for comparison
    x_hirai = result_hirai[0]
    pdf_hirai = result_hirai[1]
    
    x_vogelsang = result_vogelsang[0]
    pdf_vogelsang = result_vogelsang[1]
    
    # Only compare in region where both have significant values (x > 0.01)
    mask = x_hirai > 0.01
    
    pdf_vogelsang_interp = np.interp(x_hirai[mask], x_vogelsang, pdf_vogelsang)
    
    # Check that results agree within 10% (different numerical methods)
    np.testing.assert_allclose(
        pdf_hirai[mask], pdf_vogelsang_interp,
        rtol=0.10,  # 10% relative tolerance (methods use different approaches)
        err_msg="Hirai and Vogelsang methods give inconsistent results"
    )


# ---------------------------------------------------------------------------
# Regression tests for the issues identified in review of version 3.0.1
# ---------------------------------------------------------------------------

@pytest.fixture
def qcd():
    """QCD constants for Nf=4, Nc=3."""
    from tparton.constants import constants
    return constants(3, 4)


@pytest.mark.parametrize('s', [1, 2, 3])
@pytest.mark.parametrize('alpha_num', [False, True])
def test_lo_moment_is_pure_lo(qcd, s, alpha_num):
    """LO evolution must reproduce Eq. (26) with no NLO splitting moment.

    Version 3.0.1 zeroed `beta1` and rebound `NLO_splitting_function_moment` to
    a zero function local to `evolve()`, which had no effect on the
    module-level `evolveMoment()`. The LO evolution factor was therefore ~5%
    off the analytic LO result. This checks the factor directly in moment
    space, so it does not depend on the accuracy of the Mellin inversion.
    """
    import mpmath as mp
    from tparton import m_evolution as M

    NC, CF, Tf, beta0, _ = qcd
    if alpha_num:
        a_0 = M.alpha_S_num(4.0, 1, 91.1876 ** 2, 0.118 / 4 / mp.pi, beta0, 0)
        a_1 = M.alpha_S_num(200.0, 1, 91.1876 ** 2, 0.118 / 4 / mp.pi, beta0, 0)
    else:
        a_0 = M.alpha_S(4.0, 1, beta0, 0, 0.231)
        a_1 = M.alpha_S(200.0, 1, beta0, 0, 0.231)

    P0 = M.LO_splitting_function_moment(s, CF)
    expected = mp.power(a_1 / a_0, -2 / beta0 * P0)

    moments = [
        M.evolveMoment(s, 1, a_0, a_1, beta0, 0, eta, CF, NC, Tf, 1)
        for eta in (1, -1)
    ]
    for got in moments:
        assert abs(float(mp.re(got) / expected - 1)) < 1e-13
    # eta enters only through the NLO moment, so both branches must agree at LO
    assert moments[0] == moments[1]


def test_lo_call_does_not_perturb_later_nlo_calls(qcd):
    """An LO call must not leave shared state that changes later NLO results."""
    import mpmath as mp
    from tparton import m_evolution as M

    NC, CF, Tf, beta0, beta1 = qcd
    a_0 = M.alpha_S(4.0, 2, beta0, beta1, 0.231)
    a_1 = M.alpha_S(200.0, 2, beta0, beta1, 0.231)

    before = M.evolveMoment(2, 1, a_0, a_1, beta0, beta1, -1, CF, NC, Tf, 2)
    M.evolveMoment(2, 1, M.alpha_S(4.0, 1, beta0, 0, 0.231),
                   M.alpha_S(200.0, 1, beta0, 0, 0.231),
                   beta0, 0, -1, CF, NC, Tf, 1)
    after = M.evolveMoment(2, 1, a_0, a_1, beta0, beta1, -1, CF, NC, Tf, 2)
    assert before == after


def test_lo_differs_from_nlo_through_public_evolve(input_pdf):
    """The LO/NLO distinction must reach the public `evolve()` interface.

    Exercises the full inverse Mellin path, and checks that the two
    continuation branches coincide at LO as they must.
    """
    from tparton.m_evolution import evolve

    kw = dict(Q0_2=4.0, Q2=200.0, l_QCD=0.231, n_f=4, n_x=8,
              alpha_num=False, degree=5)
    lo = evolve(input_pdf, order=1, morp='minus', **kw)
    nlo = evolve(input_pdf, order=2, morp='minus', **kw)
    lo_plus = evolve(input_pdf, order=1, morp='plus', **kw)

    assert not np.allclose(lo[1], nlo[1])
    np.testing.assert_array_equal(lo[1], lo_plus[1])


@pytest.mark.parametrize('Q0_2, Q2', [
    (4.0, 200.0),      # upward
    (200.0, 4.0),      # downward
    (1e4, 2e4),        # entirely above the coupling reference scale
    (4.0, 20.0),       # entirely below the coupling reference scale
    (4.0, 1e5),        # straddling the coupling reference scale
])
@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('alpha_num', [False, True])
def test_direct_method_runs_in_both_directions(input_pdf, Q0_2, Q2, order, alpha_num):
    """Direct evolution must work upward and downward in Q².

    Version 3.0.1 built the coupling time series assuming an increasing Euler
    grid, so downward evolution raised `ValueError: The values in t must be
    monotonically increasing or monotonically decreasing`. It also computed the
    numerical coupling before testing `alpha_num`, so the failure occurred even
    with the analytic coupling selected.
    """
    from tparton.t_evolution import evolve

    result = evolve(input_pdf, Q0_2=Q0_2, Q2=Q2, l_QCD=0.231, n_f=4,
                    n_t=10, n_z=30, order=order, alpha_num=alpha_num,
                    logScale=True)
    assert result.shape == (2, len(input_pdf))
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize('Q0_2, Q2', [(200.0, 4.0), (4.0, 200.0)])
def test_numerical_coupling_aligns_with_euler_grid(qcd, Q0_2, Q2):
    """Each `ts[i]` must carry the coupling integrated to that same `ts[i]`.

    Compared against independent, pointwise integrations from the reference
    scale, which is the check that a plain sort of `ts` would fail.
    """
    from scipy.integrate import odeint

    _, _, _, beta0, beta1 = qcd
    n_t = 12
    ts = np.linspace(np.log(Q0_2), np.log(Q2), n_t + 1)[:-1]
    t_ref, a0 = np.log(91.1876 ** 2), 0.118 / 4 / np.pi
    ode = lambda t, a: -beta0 * a * a - beta1 * a * a * a

    reference = np.array([
        odeint(ode, a0, [t_ref, t], tfirst=True).flatten()[-1] * 2 for t in ts
    ])

    got = np.empty_like(ts)
    below = ts < t_ref
    for mask, direction in ((below, -1.0), (~below, 1.0)):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(direction * ts[idx])]
        got[idx] = odeint(
            ode, a0, np.concatenate(([t_ref], ts[idx])), tfirst=True
        ).flatten()[1:] * 2

    np.testing.assert_allclose(got, reference, rtol=1e-10)


def test_round_trip_converges_with_finer_time_grid():
    """Evolving up then back down must approach the input as n_t grows.

    Forward Euler is first order, so this checks the trend rather than an exact
    round trip.
    """
    from tparton.t_evolution import evolve

    x = np.linspace(0, 1, 40)
    pdf = np.stack((x, x * (1 - x) ** 3)).T
    kw = dict(l_QCD=0.231, n_f=4, n_z=400, alpha_num=False, logScale=True)

    errors = []
    for n_t in (50, 200):
        up = evolve(pdf, Q0_2=4.0, Q2=20.0, n_t=n_t, **kw)
        back = evolve(np.stack(up).T, Q0_2=20.0, Q2=4.0, n_t=n_t, **kw)
        mask = x > 0.01
        errors.append(np.max(np.abs(back[1][mask] - pdf[:, 1][mask]))
                      / np.max(np.abs(pdf[:, 1])))

    assert errors[1] < errors[0] / 2
    assert errors[0] < 1e-2


@pytest.mark.parametrize('method', ['t', 'm'])
def test_all_documented_input_formats_agree(method):
    """(N,), (N,1) and (N,2) inputs must give identical results.

    Version 3.0.1 raised `IndexError` on 1D input and failed on single-column
    input; only the two-column `[x, x*q(x)]` format worked.
    """
    if method == 't':
        from tparton.t_evolution import evolve
        kw = dict(n_t=8, n_z=40)
    else:
        from tparton.m_evolution import evolve
        kw = dict(n_x=6, degree=5)

    x = np.linspace(0, 1, 40)
    values = x * (1 - x) ** 3
    common = dict(Q0_2=4.0, Q2=20.0, l_QCD=0.231, n_f=4, alpha_num=False, **kw)

    from_1d = evolve(values, **common)
    from_column = evolve(values.reshape(-1, 1), **common)
    from_pairs = evolve(np.stack((x, values)).T, **common)

    np.testing.assert_array_equal(from_1d, from_column)
    np.testing.assert_array_equal(from_1d, from_pairs)


def test_unsupported_input_shape_is_rejected():
    """An input that is neither 1D nor one/two columns must raise clearly."""
    from tparton.constants import split_pdf_input

    with pytest.raises(ValueError, match=r'shape \(N,\), \(N, 1\) or \(N, 2\)'):
        split_pdf_input(np.zeros((5, 3)))


@pytest.mark.parametrize('method, expected', [
    ('t', (2, 40)),     # direct method evolves the input grid in place
    ('m', (2, 8)),      # moment method resamples onto n_x + 2 points
])
def test_documented_output_shape_and_indexing(method, expected):
    """`result[0]` must be the x grid and `result[1]` the evolved values."""
    if method == 't':
        from tparton.t_evolution import evolve
        kw = dict(n_t=8, n_z=40)
    else:
        from tparton.m_evolution import evolve
        kw = dict(n_x=6, degree=5)

    x = np.linspace(0, 1, 40)
    result = evolve(np.stack((x, x * (1 - x) ** 3)).T, Q0_2=4.0, Q2=20.0,
                    l_QCD=0.231, n_f=4, alpha_num=False, **kw)

    assert result.shape == expected
    x_out, xf_out = result[0], result[1]
    assert len(x_out) == len(xf_out) == expected[1]
    assert x_out[0] == 0.0 and x_out[-1] == 1.0
    assert np.all(np.diff(x_out) > 0)


def test_truncated_and_unexpanded_nlo_prescriptions_differ(qcd):
    """Quantify the prescription difference discussed in Sec. 4.1.

    The moment method uses the truncated NLO solution, Eq. (25), while the
    direct method approaches the unexpanded solution of the NLO evolution
    equation as the step size goes to zero. Both are NLO accurate but differ by
    formally higher-order terms, so they need not agree even with an exact
    Mellin inversion. The effect is sub-percent on an evolution factor, but is
    amplified in the difference of two nearby branches.
    """
    import mpmath as mp
    from tparton import m_evolution as M

    NC, CF, Tf, beta0, beta1 = qcd
    s = 2
    a_Q0 = M.alpha_S_num(4.0, 2, 91.1876 ** 2, 0.118 / 4 / mp.pi, beta0, beta1) / 4 / mp.pi
    a_Q = M.alpha_S_num(200.0, 2, 91.1876 ** 2, 0.118 / 4 / mp.pi, beta0, beta1) / 4 / mp.pi

    truncated, unexpanded = {}, {}
    for eta in (1, -1):
        P0 = M.LO_splitting_function_moment(s, CF)
        P1 = M.NLO_splitting_function_moment(s, eta, CF, NC, Tf)
        leading = mp.power(a_Q / a_Q0, -2 / beta0 * P0)
        truncated[eta] = mp.re(leading * (
            1 + 4 * (a_Q0 - a_Q) / beta0 * (P1 - beta1 * P0 / (2 * beta0))))
        unexpanded[eta] = mp.re(leading * mp.power(
            (beta0 + beta1 * a_Q) / (beta0 + beta1 * a_Q0),
            -(4 * P1 / beta1 - 2 * P0 / beta0)))

    # Sub-percent on each evolution factor
    for eta in (1, -1):
        rel = float(unexpanded[eta] / truncated[eta] - 1)
        assert 0.004 < rel < 0.005

    # An order of magnitude larger on the difference of the two branches
    d_tr = truncated[1] - truncated[-1]
    d_un = unexpanded[1] - unexpanded[-1]
    assert -0.15 < float(d_un / d_tr - 1) < -0.13


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
