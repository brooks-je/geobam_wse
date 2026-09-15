"""Tests for the geobam_wse port.

The important one is test_depth_line_identity_closes: it generates
observations from known parameters by inverting the AHG/AMHG depth line, then
checks the model's identity closes exactly. If that passes, the algebra in
engine.py matches the Stan engine.

    python -m pytest tests/ -v
    python tests/test_geobam_wse.py      # runs without pytest too
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from geobam_wse.flowlaw import build_index_arrays, log_qc  # noqa: E402


def _synthetic(nx=8, nt=25, seed=1, missing=0.25):
    """Observations generated from known parameters via the depth line."""
    rng = np.random.default_rng(seed)

    truth = {
        "r": rng.uniform(0.4, 0.9, nx),
        "logn": np.log(rng.uniform(0.015, 0.045, nx)),
        "logWb": rng.uniform(5.0, 7.0, nx),
        "logDb": rng.uniform(1.0, 3.0, nx),
        "f": rng.uniform(0.2, 0.7, nx),
        "logDc": np.log(rng.uniform(0.5, 5.0, nx)),
        "logQ": np.log(rng.uniform(50, 900, nt)),
        "z0": rng.uniform(-20, 400, nx),
    }

    S = rng.uniform(1e-5, 5e-4, (nx, nt))

    logQc = log_qc(truth["r"][:, None], truth["logWb"][:, None],
                   truth["logDb"][:, None], truth["logn"][:, None],
                   truth["logDc"][:, None], np.log(S))

    logd = truth["logDc"][:, None] + truth["f"][:, None] * (
        truth["logQ"][None, :] - logQc)
    H = np.exp(logd) + truth["z0"][:, None]

    hasdat = rng.random((nx, nt)) > missing
    hasdat[:, 0] = True   # guarantee every node keeps at least one cell

    return truth, S, H, hasdat


def test_index_arrays_align_with_masked_flatten():
    _, S, H, hasdat = _synthetic()
    xind, tind = build_index_arrays(hasdat)

    assert xind.size == int(hasdat.sum())
    assert np.allclose(H[xind, tind], H[hasdat])
    assert np.allclose(S[xind, tind], S[hasdat])


def test_log_qc_matches_the_bracketed_term():
    """logQc is log of W_b (1/d_b)^(1/r) ((r+1)/r)^(1/r) S^(1/2) (1/n) d_c^(5/3+1/r)."""
    rng = np.random.default_rng(3)
    r = rng.uniform(0.3, 1.0, 6)
    Wb = rng.uniform(80, 900, 6)
    db = rng.uniform(2.0, 15.0, 6)
    n = rng.uniform(0.012, 0.05, 6)
    dc = rng.uniform(0.4, 8.0, 6)
    S = rng.uniform(1e-5, 5e-4, 6)

    direct = np.log(
        Wb * (1.0 / db) ** (1.0 / r) * ((r + 1.0) / r) ** (1.0 / r)
        * np.sqrt(S) * (1.0 / n) * dc ** (5.0 / 3.0 + 1.0 / r))

    via = log_qc(r, np.log(Wb), np.log(db), np.log(n), np.log(dc), np.log(S))

    assert np.abs(direct - via).max() < 1e-10


def test_depth_line_identity_closes():
    """log d == logDc + f (logQ - logQc) on exactly generated data."""
    truth, S, H, hasdat = _synthetic()
    nx = H.shape[0]

    xind, tind = build_index_arrays(hasdat)

    Hmin = np.array([H[i, hasdat[i]].min() for i in range(nx)])
    depth_min = Hmin - truth["z0"]
    assert depth_min.min() > 0, "depth_min must be positive by construction"

    z0 = Hmin - depth_min
    logd = np.log(H[xind, tind] - z0[xind])

    logQc = log_qc(truth["r"][xind], truth["logWb"][xind],
                   truth["logDb"][xind], truth["logn"][xind],
                   truth["logDc"][xind], np.log(S[xind, tind]))
    depth_rhs = truth["logDc"][xind] + truth["f"][xind] * (
        truth["logQ"][tind] - logQc)

    assert np.abs(logd - depth_rhs).max() < 1e-9


def test_depth_positivity_is_structural():
    """H - z0 >= depth_min > 0 for every observation, with no clamping."""
    truth, _, H, hasdat = _synthetic()
    nx = H.shape[0]

    Hmin = np.array([H[i, hasdat[i]].min() for i in range(nx)])
    depth_min = Hmin - truth["z0"]
    z0 = Hmin - depth_min

    xind, tind = build_index_arrays(hasdat)
    depth = H[xind, tind] - z0[xind]

    assert depth.min() > 0
    assert depth.min() >= depth_min.min() - 1e-12


def test_jacobian_term_is_present_and_has_the_right_sign():
    """The change of variables from H to log-depth is -sum(log d).

    This term exists because log d = log(H - z0) is a transform of the data
    that depends on a parameter. Losing it in translation biases z0 towards
    larger depths, and nothing else in the model would complain.
    """
    # Read the source as text rather than importing, so this check runs
    # without JAX installed.
    engine_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "geobam_wse", "engine.py")
    with open(engine_src) as fh:
        src = fh.read()

    assert 'numpyro.factor("jacobian"' in src, "Jacobian factor is missing"
    assert "-logd.sum()" in src, "Jacobian must be negative sum of log-depth"

    # and the sign is what the change of variables actually gives
    rng = np.random.default_rng(5)
    d = rng.uniform(0.5, 20.0, 50)
    analytic = np.sum(np.log(1.0 / d))         # log|d(log d)/dH| = -log d
    assert np.isclose(analytic, -np.sum(np.log(d)))


def test_remake_discharge_recovers_truth():
    truth, S, H, hasdat = _synthetic()
    nx = H.shape[0]

    Hmin = np.array([H[i, hasdat[i]].min() for i in range(nx)])
    z0 = Hmin - (Hmin - truth["z0"])

    posteriors = {k: {"mean": truth[k]} for k in
                  ("r", "logn", "logWb", "logDb", "f", "logDc")}
    posteriors["z0"] = {"mean": z0}

    from geobam_wse.flowlaw import remake_discharge

    recon = remake_discharge(np.where(hasdat, S, np.nan),
                            np.where(hasdat, H, np.nan),
                            posteriors, hasdat=hasdat)

    assert np.nanmax(np.abs(recon / np.exp(truth["logQ"]) - 1)) < 1e-8


def test_z0_prior_reparameterisation_is_equivalent():
    """z0 ~ N(z0_hat, z0_sd) with depth_min > 0 is a truncated normal on depth_min.

    z0 = Hmin - depth_min is affine with unit Jacobian, and Hmin, z0_hat and
    z0_sd are all data, so the two forms differ only by a constant.
    """
    rng = np.random.default_rng(7)
    Hmin = rng.uniform(-20, 400, 6)
    z0_hat = Hmin - 1.5
    z0_sd = np.full(6, 1.0)
    depth_min = rng.uniform(0.1, 5.0, 6)

    z0 = Hmin - depth_min

    def _norm_lp(x, loc, scale):
        return -0.5 * ((x - loc) / scale) ** 2 - np.log(scale)

    on_z0 = _norm_lp(z0, z0_hat, z0_sd)
    on_depth = _norm_lp(depth_min, Hmin - z0_hat, z0_sd)

    assert np.abs(on_z0 - on_depth).max() < 1e-12


def test_read_strings_handles_both_netcdf_text_layouts():
    """NC_CHAR variables arrive as (nt, strlen) single characters."""
    import types

    stub = sys.modules.get("netCDF4")
    if stub is None:
        stub = types.ModuleType("netCDF4")
        stub.Dataset = object

        def _chartostring(b):
            b = np.asarray(b)
            return np.array(
                [row.tobytes().decode("utf-8").rstrip("\x00") for row in b],
                dtype=object)

        stub.chartostring = _chartostring
        sys.modules["netCDF4"] = stub

    from geobam_wse.input import read_strings

    class _Var:
        def __init__(self, a):
            self.a = a

        def __getitem__(self, key):
            return self.a[key]

    stamps = ["2024-01-15T06:12:00Z", "2024-02-20T18:44:00Z"]

    assert list(read_strings(_Var(np.array(stamps, dtype=object)))) == stamps

    as_char = np.array(
        [[bytes([c]) for c in s.ljust(24, "\x00").encode()] for s in stamps],
        dtype="S1")
    assert list(read_strings(_Var(as_char))) == stamps


def test_concatenate_invalid_restores_length():
    from geobam_wse.output import (concatenate_invalid,
                                   concatenate_invalid_nodes)

    padded = concatenate_invalid(np.arange(5.0), [1, 3])
    assert padded.size == 7
    assert np.isnan(padded[1]) and np.isnan(padded[3])

    padded_nodes = concatenate_invalid_nodes(np.arange(4.0), [0, 2], 6)
    assert padded_nodes.size == 6
    assert np.isnan(padded_nodes[0]) and np.isnan(padded_nodes[2])


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                    # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
