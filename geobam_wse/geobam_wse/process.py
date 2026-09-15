"""Assemble model inputs and run the engine.

Port of process.R.
"""

import os

import numpy as np

from .engine import run_sampler
from .flowlaw import build_index_arrays

# Reach-constant priors, carried over unchanged from process.R.
R_HAT, R_SD = 0.55, 0.15
R_LOWER, R_UPPER = 0.3, 1.0

LOGWB_HAT, LOGWB_SD = 6.44, 1.22
LOGWB_LOWER, LOGWB_UPPER = 4.4, 9.95

LOGDB_HAT, LOGDB_SD = 2.36, 0.83
LOGDB_LOWER, LOGDB_UPPER = -0.03, 3.85

LOGN_HAT, LOGN_SD = np.log(0.03), 0.1
LOGN_LOWER, LOGN_UPPER = np.log(0.01), np.log(0.05)

# Depth-model priors. f is the AHG depth exponent, d_c the depth scale.
F_HAT, F_SD = 0.45, 0.10
F_LOWER, F_UPPER = 0.05, 0.90

LOGDC_HAT, LOGDC_SD = np.log(1.5), 0.50
LOGDC_LOWER, LOGDC_UPPER = np.log(0.2), np.log(20.0)

# Bed-elevation prior scale, in metres.
Z0_SD = 1.0

# Structural error of the flow law. NOTE: the likelihood is on log-DEPTH, not
# log-Q. A log-Q error of sigma maps to roughly f * sigma in log-depth, so at
# F_HAT = 0.45 the historical 0.26 is about 2.2x wider than intended. Left at
# 0.26 to match the R module; override with GEOBAM_WSE_SIGMA_MAN to test.
SIGMA_MAN = 0.26

ITER = 2000


def _env_num(name, default):
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def build_model_data(data, width_gates_hasdat=False):
    """Turn the screened observations into the flat dict the engine consumes.

    width_gates_hasdat mirrors the R module's `Wobs > 0` condition in hasdat.
    It defaults to False: width does not appear in the flow law (the Stan
    engine declares Wobs and never reads it), and input.R already stopped
    letting width drop whole nodes and timesteps, so gating individual cells
    on it only discards usable observations. Set True to reproduce the R
    module's cell selection exactly when diffing.
    """
    Hobs = np.asarray(data["swot_data"]["wse"], dtype=float)
    Sobs = np.asarray(data["swot_data"]["slope2"], dtype=float)
    Wobs = np.asarray(data["swot_data"]["width"], dtype=float)

    nx, nt = Hobs.shape

    hasdat = np.isfinite(Sobs) & (Sobs > 0) & np.isfinite(Hobs)
    if width_gates_hasdat:
        hasdat &= np.isfinite(Wobs) & (Wobs > 0)

    ntot = int(hasdat.sum())
    if ntot < 3:
        raise ValueError("fewer than 3 observations have both slope and WSE")

    xind, tind = build_index_arrays(hasdat)

    Hobsvec = Hobs[xind, tind]
    Sobsvec = Sobs[xind, tind]
    logSobsvec = np.log(Sobsvec)

    # Per-node minimum of the WSE cells hasdat marks usable. The engine
    # derives z0 = Hmin - depth_min from this, so it has to be computed over
    # exactly the same set or the prior and the parameterisation disagree.
    Hmin = np.full(nx, np.nan)
    for i in range(nx):
        row = Hobs[i, hasdat[i]]
        if row.size:
            Hmin[i] = row.min()

    if not np.isfinite(Hmin).all():
        n_missing = int((~np.isfinite(Hmin)).sum())
        fallback = (np.nanmin(Hmin) if np.isfinite(Hmin).any() else 0.0)
        print(f"  {n_missing} node(s) have no usable observation; anchoring "
              f"their z0 prior at the reach-wide minimum WSE", flush=True)
        Hmin[~np.isfinite(Hmin)] = fallback

    logDc_hat = np.full(nx, LOGDC_HAT)

    # z0 prior: the lowest observed water surface, less a nominal depth d_c.
    z0_hat = Hmin - np.exp(logDc_hat)
    z0_sd = np.full(nx, _env_num("GEOBAM_WSE_Z0_SD", Z0_SD))

    sigma_man = _env_num("GEOBAM_WSE_SIGMA_MAN", SIGMA_MAN)

    print(f"z0 prior: {np.exp(LOGDC_HAT):.2f} m below each node's minimum "
          f"WSE, sd {z0_sd[0]:.2f} m | sigma_man {sigma_man:.3f} "
          f"(log-depth scale)", flush=True)

    q = data["sos_data"]["Q_priors"]

    return {
        "nx": nx, "nt": nt, "ntot": ntot,
        "xind": xind, "tind": tind,
        "Hobsvec": Hobsvec,
        "logSobsvec": logSobsvec,
        "Hmin": Hmin,
        "sigma_vec_man": np.full(ntot, sigma_man),

        "logQ_hat": np.asarray(q["logQ_hat"], dtype=float),
        "logQ_sd": np.full(nt, float(q["logQ_sd"])),
        "lowerbound_logQ": float(q["lowerbound_logQ"]),
        "upperbound_logQ": float(q["upperbound_logQ"]),

        "r_hat": np.full(nx, R_HAT), "r_sd": np.full(nx, R_SD),
        "lowerbound_r": R_LOWER, "upperbound_r": R_UPPER,

        "logWb_hat": np.full(nx, LOGWB_HAT), "logWb_sd": np.full(nx, LOGWB_SD),
        "lowerbound_logWb": LOGWB_LOWER, "upperbound_logWb": LOGWB_UPPER,

        "logDb_hat": np.full(nx, LOGDB_HAT), "logDb_sd": np.full(nx, LOGDB_SD),
        "lowerbound_logDb": LOGDB_LOWER, "upperbound_logDb": LOGDB_UPPER,

        "logn_hat": np.full(nx, LOGN_HAT), "logn_sd": np.full(nx, LOGN_SD),
        "lowerbound_logn": LOGN_LOWER, "upperbound_logn": LOGN_UPPER,

        "f_hat": np.full(nx, F_HAT), "f_sd": np.full(nx, F_SD),
        "lowerbound_f": F_LOWER, "upperbound_f": F_UPPER,

        "logDc_hat": logDc_hat, "logDc_sd": np.full(nx, LOGDC_SD),
        "lowerbound_logDc": LOGDC_LOWER, "upperbound_logDc": LOGDC_UPPER,

        "z0_hat": z0_hat, "z0_sd": z0_sd,

        "iter": ITER,
    }


def process_data(data, seed=0, width_gates_hasdat=False):
    """Run the model for one reach."""
    model_data = build_model_data(data, width_gates_hasdat)

    posteriors, logQ_draws = run_sampler(model_data, seed=seed)

    # Q is summarised in linear space, so exponentiate the draws before taking
    # moments rather than after.
    q_draws = np.exp(logQ_draws)
    posterior_Q = q_draws.mean(axis=0)
    posterior_Q_sd = q_draws.std(axis=0, ddof=1)

    return {
        "posterior_Q": posterior_Q,
        "posterior_Q_sd": posterior_Q_sd,
        "posteriors": posteriors,
    }
