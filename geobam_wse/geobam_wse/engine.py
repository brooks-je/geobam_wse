"""NumPyro engine for geobam_wse.

Flow law (AHG/AMHG depth formulation):

    Q = ((H - z0)/d_c)^(1/f) * [ W_b (1/d_b)^(1/r) ((r+1)/r)^(1/r)
                                 S^(1/2) (1/n) d_c^(5/3 + 1/r) ]

The bracketed term is the centre-point discharge Qc, the discharge at which
depth equals d_c. Solving for log-depth gives the line the likelihood is
written on:

    logQc = logWb - (1/r) logDb + (1/r)(log(r+1) - log r)
            + (1/2) logS - logn + (5/3 + 1/r) logDc

    log d = logDc + f (logQ - logQc)

Bed elevation is parameterised through depth_min, the depth at the lowest
observed water surface:

    z0_i = Hmin_i - depth_min_i,   depth_min_i > 0

so H - z0 >= depth_min > 0 for every observation by construction.

Two things here have no counterpart in geobam_wse's sibling module and are
easy to lose in translation:

  1. The likelihood is on OBSERVED log-depth, not on log-Q. log d is a
     transform of the data H that depends on a parameter (z0), so keeping
     this a density over H needs the change-of-variables term
     d/dH log(H - z0) = 1/(H - z0), i.e. `-sum(logd)`. That is the
     `target += -sum(logd)` line in the Stan engine and the explicit
     `numpyro.factor("jacobian", ...)` below. Drop it and z0 is biased.

  2. The z0 prior is written on the DERIVED z0, not on depth_min. Because
     z0 = Hmin - depth_min is affine with unit Jacobian and Hmin, z0_hat and
     z0_sd are all data, `z0 ~ normal(z0_hat, z0_sd)` with depth_min > 0 is
     exactly a normal on depth_min located at Hmin - z0_hat and truncated at
     zero. That is how it is expressed here.
"""

import os
import time

import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from .flowlaw import build_index_arrays, remake_discharge  # noqa: F401
# re-exported so callers can keep importing them from engine

# Parameters reported per node, in the order the output module expects.
NODE_PARAMS = ("r", "logn", "logWb", "logDb", "f", "logDc", "z0")


def _env_num(name, default, cast=float):
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return cast(float(raw))
    except (TypeError, ValueError):
        return default


def _truncated(name, loc, scale, low, high, size):
    return numpyro.sample(
        name,
        dist.TruncatedNormal(loc=loc, scale=scale,
                             low=float(low), high=float(high))
        .expand([size]).to_event(1),
    )


def geobam_wse_model(data):
    """The AHG/AMHG depth model as a NumPyro model."""
    nx = int(data["nx"])
    nt = int(data["nt"])

    xind = data["xind"]
    tind = data["tind"]

    Hobsvec = data["Hobsvec"]
    logSobsvec = data["logSobsvec"]
    sigma_vec = data["sigma_vec_man"]
    Hmin = data["Hmin"]

    # --- Parameters -------------------------------------------------------
    logQ = _truncated("logQ", data["logQ_hat"], data["logQ_sd"],
                      data["lowerbound_logQ"], data["upperbound_logQ"], nt)
    r = _truncated("r", data["r_hat"], data["r_sd"],
                   data["lowerbound_r"], data["upperbound_r"], nx)
    logWb = _truncated("logWb", data["logWb_hat"], data["logWb_sd"],
                       data["lowerbound_logWb"], data["upperbound_logWb"], nx)
    logn = _truncated("logn", data["logn_hat"], data["logn_sd"],
                      data["lowerbound_logn"], data["upperbound_logn"], nx)
    logDb = _truncated("logDb", data["logDb_hat"], data["logDb_sd"],
                       data["lowerbound_logDb"], data["upperbound_logDb"], nx)
    f = _truncated("f", data["f_hat"], data["f_sd"],
                   data["lowerbound_f"], data["upperbound_f"], nx)
    logDc = _truncated("logDc", data["logDc_hat"], data["logDc_sd"],
                       data["lowerbound_logDc"], data["upperbound_logDc"], nx)

    # See note 2 in the module docstring: this is the z0 prior, re-expressed.
    depth_min = numpyro.sample(
        "depth_min",
        dist.TruncatedNormal(loc=Hmin - data["z0_hat"],
                             scale=data["z0_sd"], low=0.0)
        .expand([nx]).to_event(1),
    )

    # --- Derived ----------------------------------------------------------
    z0 = numpyro.deterministic("z0", Hmin - depth_min)

    r_c = r[xind]
    invr = 1.0 / r_c

    d_obs = Hobsvec - z0[xind]      # >= depth_min > 0 by construction
    logd = jnp.log(d_obs)

    # logQc: the bracketed centre-point discharge, per observation.
    logQc = (logWb[xind]
             - invr * logDb[xind]
             + invr * (jnp.log(r_c + 1.0) - jnp.log(r_c))
             + 0.5 * logSobsvec
             - logn[xind]
             + (5.0 / 3.0 + invr) * logDc[xind])

    # The AHG/AMHG depth line.
    depth_rhs = logDc[xind] + f[xind] * (logQ[tind] - logQc)

    # Likelihood on observed log-depth. logd depends on depth_min, so this
    # cannot be an `obs=` site.
    numpyro.factor(
        "depth_likelihood",
        dist.Normal(depth_rhs, sigma_vec).log_prob(logd).sum(),
    )

    # See note 1: change of variables from H to log-depth.
    numpyro.factor("jacobian", -logd.sum())


def run_sampler(data, seed=0):
    """Sample the posterior. Returns (summary_dict, logQ_draws)."""
    iter_total = int(_env_num("GEOBAM_WSE_ITER", data.get("iter", 2000)))
    warmup = int(_env_num("GEOBAM_WSE_WARMUP", max(500, int(0.4 * iter_total))))
    num_samples = max(1, iter_total - warmup)
    chains = int(_env_num("GEOBAM_WSE_CHAINS", 3))
    max_td = int(_env_num("GEOBAM_WSE_MAX_TREEDEPTH", 10))
    adapt_delta = _env_num("GEOBAM_WSE_ADAPT_DELTA", 0.8)
    dense = bool(_env_num("GEOBAM_WSE_DENSE_MASS", 0))
    progress = bool(_env_num("GEOBAM_WSE_PROGRESS", 1))

    print(f"problem size: nx={data['nx']} nt={data['nt']} ntot={data['ntot']}",
          flush=True)
    print(f"sampler: warmup={warmup} samples={num_samples} chains={chains} "
          f"max_treedepth={max_td} target_accept={adapt_delta} "
          f"dense_mass={dense}", flush=True)

    # Start depth_min at its prior location rather than wherever the
    # unconstrained default lands, which for a deep river puts z0 far above
    # the bed and starts the chain in a badly curved region.
    d0 = np.asarray(data["Hmin"] - data["z0_hat"], dtype=float)
    d0[~np.isfinite(d0) | (d0 <= 0)] = 1.0
    init_vals = {"depth_min": jnp.asarray(d0)}

    kernel = NUTS(
        geobam_wse_model,
        max_tree_depth=max_td,
        target_accept_prob=adapt_delta,
        dense_mass=dense,
        init_strategy=init_to_value(values=init_vals),
    )

    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=num_samples,
        num_chains=chains,
        chain_method="parallel" if chains > 1 else "sequential",
        progress_bar=progress,
    )

    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), data,
             extra_fields=("diverging", "num_steps"))
    print(f"sampling took {(time.time() - t0) / 60.0:.1f} min", flush=True)

    _report_diagnostics(mcmc, max_td)

    samples = mcmc.get_samples(group_by_chain=False)
    return _summarise(samples), np.asarray(samples["logQ"])


def _report_diagnostics(mcmc, max_td):
    """Print the diagnostics that explain a slow or untrustworthy run."""
    try:
        extra = mcmc.get_extra_fields()
        div = int(np.sum(np.asarray(extra["diverging"])))
        steps = np.asarray(extra["num_steps"])
        n = steps.size
        saturated = int(np.sum(steps >= (2 ** max_td) - 1))

        summary = numpyro.diagnostics.summary(
            mcmc.get_samples(group_by_chain=True), group_by_chain=True)
        rhats, neffs = [], []
        for stats in summary.values():
            rhats.append(np.nanmax(stats["r_hat"]))
            neffs.append(np.nanmin(stats["n_eff"]))

        print(f"diagnostics: {div}/{n} divergent, {saturated}/{n} at "
              f"max_treedepth, max Rhat {np.nanmax(rhats):.3f}, "
              f"min n_eff {np.nanmin(neffs):.0f}", flush=True)

        if saturated > 0.2 * n:
            print("  -> treedepth saturation dominates runtime; "
                  "raise GEOBAM_WSE_WARMUP", flush=True)
        if div > 0:
            print("  -> divergences: most likely the logDc / depth_min ridge "
                  "(both are depth scales anchored to the same observations). "
                  "Tighten z0_sd or raise GEOBAM_WSE_ADAPT_DELTA", flush=True)
    except Exception as exc:                      # diagnostics are never fatal
        print(f"diagnostics unavailable: {exc}", flush=True)


def _summarise(samples):
    """Posterior mean and sd for the reported parameters.

    The sd convention matches the R module exactly -- one scalar per
    parameter, the mean of the per-node posterior variances -- so output from
    the two implementations can be diffed directly. It is a known wart, not a
    design choice; see PORTING_NOTES.md.
    """
    out = {}
    for name in NODE_PARAMS + ("logQ", "depth_min"):
        if name not in samples:
            continue
        draws = np.asarray(samples[name])
        out[name] = {
            "mean": draws.mean(axis=0),
            "sd": float(np.mean(draws.std(axis=0, ddof=1) ** 2)),
        }
    return out
