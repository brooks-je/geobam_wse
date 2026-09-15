# R to Python porting notes — geobam_wse

## What carried over from the neobam_wse port, and what didn't

`input.py`, `output.py`, the package layout, the Dockerfile and the sampler
plumbing are shared with the `neobam_wse` port. **The engine is not.** These
two modules use different flow laws, and `geobam_wse` has two features its
sibling does not:

1. **The likelihood is on observed log-depth, not on log-Q.** In
   `neobam_wse` both sides of the likelihood are log-Q quantities and the data
   enter through `logS`. Here the left-hand side is `log(H - z0)` — a
   transform of the data that depends on a parameter.

2. **That transform needs a Jacobian.** `d/dH log(H - z0) = 1/(H - z0)`, so
   keeping this a density over `H` requires `-sum(log d)`. It is the
   `target += -sum(logd)` line in the Stan engine and an explicit
   `numpyro.factor("jacobian", -logd.sum())` here. Lose it and `z0` biases
   towards larger depths with nothing else in the model complaining, which is
   why there is a test guarding it.

## Translation decisions

**Neither likelihood term can be an `obs=` site.** Both `logd` and
`depth_rhs` depend on parameters. NumPyro's `obs=` requires data, so both are
`numpyro.factor`. Reaching for `dist.Normal(depth_rhs, sigma, obs=logd)` would
silently give a different model.

**The z0 prior is re-expressed on `depth_min`.** The Stan engine declares
`depth_min` with `lower=0` and no prior of its own, then writes
`z0 ~ normal(z0_hat, z0_sd)` on the derived quantity. Since
`z0 = Hmin - depth_min` is affine with unit Jacobian and `Hmin`, `z0_hat` and
`z0_sd` are all data, that is exactly a normal on `depth_min` located at
`Hmin - z0_hat` and truncated at zero. `test_z0_prior_reparameterisation_is_equivalent`
checks the two log-densities agree to 1e-12.

**`<lower=, upper=>` plus a normal prior is a truncated normal.** All seven
bounded parameters use `dist.TruncatedNormal`; NumPyro handles the support
transform itself.

**The ragged helpers are gone.** `ragged_vec`, `ragged_row` and `ragged_col`
looped over all `nx * nt` cells with a branch per cell, seven times per
gradient evaluation. `build_index_arrays()` computes `(xind, tind)` once and
the model indexes directly. This is where most of the speedup comes from,
independent of backend.

## Changes from the R module

- **`hasdat` no longer requires width by default.** `input.R` had already
  been changed so width does not drop whole nodes and timesteps, but
  `process.R` still gated individual cells on `Wobs > 0` — a half-applied
  change. Width appears nowhere in the flow law (the Stan engine declares
  `Wobs` and never reads it), so it now gates nothing. Pass
  `--width-gates-hasdat` to restore the old behaviour when diffing.

- **`iter = 200, warmup = 100` were left in `process.R`** from a smoke test.
  The port defaults to `iter = 2000` with `warmup = max(500, 0.4*iter)` and
  makes both overridable by environment variable.

- **`q_sd` is computed in linear space and written on the `nt` dimension.**
  The R module wrote `exp(posteriors$logQ$sd)` into a scalar variable, where
  `$sd` was itself the mean of per-timestep log-space variances. That was
  neither an SD nor per-timestep. This matches the `neobam_wse` port, so both
  modules now write the same shape.

- **Posteriors are padded over dropped nodes.** `output.R` declared `nx` from
  the full SoS node list but wrote posteriors of the post-screening length.

- **The SoS retry loop terminates.** `input.R` decremented `tries` inside the
  `tryCatch` handler, which is a closure, so the outer counter never moved.

- **The output mount is checked before sampling**, and NetCDF array
  orientation is verified against `nx`/`nt` rather than assumed.

- **`time_str` handles `NC_CHAR` as well as `NC_STRING`** — the fix from the
  `neobam_wse` run, carried over.

## Still open, carried over from the R module

- **`sigma_man = 0.26` is on the wrong scale.** It was the flow law's
  structural error in log-Q. This likelihood is in log-depth, and
  `log d = logDc + f(logQ - logQc)`, so a log-Q error of σ maps to `f·σ` in
  log-depth. At `f_hat = 0.45` the effective error is roughly 2.2× wider than
  intended. Left at 0.26 to match the R module; try
  `GEOBAM_WSE_SIGMA_MAN=0.12` and compare.

- **`z0_hat` and `logDc_hat` are the same number.** `z0_hat = Hmin - exp(logDc_hat)`
  with `logDc_hat = log(1.5)`, so the bed-elevation prior and the AHG depth
  scale are both anchored to 1.5 m. `d_c` and `depth_min` are two depth scales
  fitted to the same observations, and tying their priors together makes the
  ridge between them worse. If divergences show up, this is the first thing to
  separate.

- **`logn`, `logWb` and `logDb` enter `logQc` as one additive per-node
  constant.** Nothing in the data separates them, only their priors do.

- **`f` and `logDc` priors are reach-constant placeholders** (0.45 and
  log(1.5) for every node on every reach).

## Not run here

I could not install JAX, NumPyro or netCDF4 in this environment, so the
sampler has not been executed end to end. Everything that does not need them
is tested and passing: the depth-line identity, `logQc` against the bracketed
product written out longhand, the Jacobian sign, the z0 reparameterisation,
index alignment, depth positivity, discharge reconstruction, the `NC_CHAR`
string fix and the pad-back helpers.

First real run, in order:

1. `python -m pytest tests/ -v`
2. One reach with `GEOBAM_WSE_ITER=400 GEOBAM_WSE_WARMUP=200 GEOBAM_WSE_CHAINS=1`
   — confirm it completes and writes finite `z0`, `f` and `logDc`.
3. Full settings, then compare `q`, `z0` and `f` against the R module for the
   same reach. Expect overlap within uncertainty, not equality — different
   sampler, adaptation and RNG.
