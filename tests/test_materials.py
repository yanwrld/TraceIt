"""
test spec for traceit.materials; reference values are verified independently

written by claude opus 5.5
"""

import numpy as np
import pytest

from traceit.materials import Conductor, Dielectric
from traceit.units import GHZ, MHZ, UM

FR4 = Dielectric("test FR-4", dk=4.0, df=0.02, f_ref=1 * GHZ)


# ---------------- Djordjevic-Sarkar dielectric ----------------
def test_ds_fit_parameters():
    assert FR4.eps_inf == pytest.approx(3.7408, abs=1e-4)
    assert FR4.delta_eps == pytest.approx(0.9419, abs=1e-4)


def test_ds_round_trip_at_f_ref():
    e = FR4.eps_r(1 * GHZ)
    assert e.real == pytest.approx(4.0, rel=1e-12)
    assert -e.imag / e.real == pytest.approx(0.02, rel=1e-12)


@pytest.mark.parametrize(
    "f, dk, df",
    [
        (1 * MHZ, 4.3532, 0.01843),
        (10 * GHZ, 3.8824, 0.01986),
        (50 * GHZ, 3.8024, 0.01703),
    ],
)
def test_ds_reference_values(f, dk, df):
    e = FR4.eps_r(f)
    assert e.real == pytest.approx(dk, abs=1e-4)
    assert -e.imag / e.real == pytest.approx(df, abs=1e-5)


def test_ds_physics():
    f = np.linspace(0, 100 * GHZ, 1001)
    e = FR4.eps_r(f)
    assert e.shape == f.shape and np.iscomplexobj(e)
    assert np.all(np.isfinite(e))  # finite at DC
    assert np.all(np.diff(e.real) < 0)  # Dk falls with frequency (causality)
    assert np.all(-e.imag[1:] > 0)  # lossy (eps'' > 0) for f > 0
    assert e.imag[0] == pytest.approx(0.0, abs=1e-6)  # ~no loss at DC


def test_ds_scalar_input():
    assert np.ndim(FR4.eps_r(1e9)) == 0


def test_ds_lossless_limit():
    air_like = Dielectric("lossless", dk=3.0, df=0.0)
    e = air_like.eps_r(np.array([0, 1e9, 1e11]))
    assert np.allclose(e, 3.0)


def test_dielectric_is_hashable():
    assert hash(FR4) == hash(Dielectric("test FR-4", dk=4.0, df=0.02, f_ref=1 * GHZ))


@pytest.mark.parametrize("dk, df", [(0.5, 0.01), (4.0, -0.01), (4.0, 1.5)])
def test_dielectric_rejects_bad_input(dk, df):
    with pytest.raises(ValueError):
        Dielectric("bad", dk=dk, df=df)


# ---------------- Conductor ----------------
CU = Conductor()


@pytest.mark.parametrize("f, delta_um", [(1 * GHZ, 2.090), (10 * GHZ, 0.661)])
def test_skin_depth(f, delta_um):
    assert CU.skin_depth(f) / UM == pytest.approx(delta_um, abs=1e-3)


def test_surface_resistance():
    assert CU.rs(10 * GHZ) == pytest.approx(0.0261, abs=1e-4)
    # Rs = 1 / (sigma * delta) identity
    f = np.array([1e8, 1e9, 1e10])
    assert np.allclose(CU.rs(f), 1 / (CU.sigma * CU.skin_depth(f)))


def test_rs_scales_as_sqrt_f():
    assert CU.rs(4e9) / CU.rs(1e9) == pytest.approx(2.0)


def test_smooth_copper_has_no_roughness_penalty():
    assert np.all(CU.rough_factor(np.array([0.0, 1e9, 1e11])) == 1.0)


def test_hammerstad_roughness():
    rough = Conductor(rq=1 * UM)
    assert rough.rough_factor(10 * GHZ) == pytest.approx(1.807, abs=1e-3)
    f = np.linspace(0, 200 * GHZ, 201)
    k = rough.rough_factor(f)
    assert k[0] == pytest.approx(1.0)  # no penalty at DC
    assert np.all(np.diff(k) >= 0)  # monotonic
    assert np.all((k >= 1) & (k < 2))  # Hammerstad saturates at 2


@pytest.mark.parametrize("sigma, rq", [(0.0, 0.0), (5.8e7, -1e-6)])
def test_conductor_rejects_bad_input(sigma, rq):
    with pytest.raises(ValueError):
        Conductor(sigma=sigma, rq=rq)
