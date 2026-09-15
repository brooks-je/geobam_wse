# geobam_wse (Python / NumPyro)

Port of the R/rstan `geobam_wse` Confluence module.

## Flow law

    Q = ((H - z0)/d_c)^(1/f) * [ W_b (1/d_b)^(1/r) ((r+1)/r)^(1/r)
                                 S^(1/2) (1/n) d_c^(5/3 + 1/r) ]

The bracketed term is the centre-point discharge `Qc` — the discharge at which
depth equals `d_c`. Solved for log-depth, which is the line the likelihood
sits on:

    logQc = logWb - (1/r) logDb + (1/r)(log(r+1) - log r)
            + (1/2) logS - logn + (5/3 + 1/r) logDc

    log d = logDc + f (logQ - logQc)

Bed elevation is parameterised through `depth_min`, the depth at the lowest
observed water surface: `z0_i = Hmin_i - depth_min_i` with `depth_min_i > 0`,
so `H - z0 >= depth_min > 0` everywhere by construction.

## Layout

| File | Replaces |
| --- | --- |
| `run_geobam_wse.py` | `run_geobam_wse.R` |
| `geobam_wse/input.py` | `input.R` |
| `geobam_wse/engine.py` | `geobam_wse_stan_engine.stan` + the rstan glue |
| `geobam_wse/flowlaw.py` | `remake_discharge()` and the ragged mapping |
| `geobam_wse/process.py` | `process.R` |
| `geobam_wse/output.py` | `output.R` |

`flowlaw.py` holds the numpy-only arithmetic so it can be imported and tested
without JAX. `output.py` imports `netCDF4` lazily for the same reason.

`config.R` and `prior_functions.R` are not ported: nothing in the module
sources either of them.

## Running

```bash
python run_geobam_wse.py -r reaches.json -i 0
```

The index is 0-based. Omit `-i` (or pass `-256`) to take it from
`AWS_BATCH_JOB_ARRAY_INDEX` instead.

```bash
docker build -t brooksje/geobam_wse:latest .
docker run --rm --memory=8g \
  -v "$MNT/input:/mnt/data/input" \
  -v "$MNT/flpe/geobam_wse:/mnt/data/output" \
  brooksje/geobam_wse:latest -r reaches.json -i 0
```

### Environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `GEOBAM_WSE_ITER` | 2000 | total iterations, warmup included |
| `GEOBAM_WSE_WARMUP` | `max(500, 0.4*iter)` | set explicitly for short runs |
| `GEOBAM_WSE_CHAINS` | 3 | one JAX host device per chain |
| `GEOBAM_WSE_MAX_TREEDEPTH` | 10 | |
| `GEOBAM_WSE_ADAPT_DELTA` | 0.8 | NumPyro's `target_accept_prob` |
| `GEOBAM_WSE_DENSE_MASS` | 0 | set to 1 for a dense mass matrix |
| `GEOBAM_WSE_PROGRESS` | 1 | set to 0 for quieter batch logs |
| `GEOBAM_WSE_SIGMA_MAN` | 0.26 | flow-law error, **log-depth** scale |
| `GEOBAM_WSE_Z0_SD` | 1.0 | bed-elevation prior sd, metres |

Smoke test — note that `WARMUP` must be set explicitly, since its default
floor of 500 would otherwise exceed a short `ITER`:

```bash
docker run --rm --memory=8g \
  --env GEOBAM_WSE_ITER=400 \
  --env GEOBAM_WSE_WARMUP=200 \
  --env GEOBAM_WSE_CHAINS=1 \
  -v "$MNT/input:/mnt/data/input" \
  -v "$MNT/flpe/geobam_wse:/mnt/data/output" \
  brooksje/geobam_wse:latest -r reaches.json -i 0
```

### Flags

`--width-gates-hasdat` restores the R module's `Wobs > 0` condition on
individual cells. Off by default — width is not in the flow law (the Stan
engine declared `Wobs` and never read it), and `input.R` already stopped
letting width drop whole nodes and timesteps. Use it only to reproduce the R
module's cell selection when diffing.

## Tests

```bash
python -m pytest tests/ -v      # or: python tests/test_geobam_wse.py
```

Nine tests, no JAX or NetCDF needed. The load-bearing ones:

- `test_depth_line_identity_closes` — generates observations by inverting the
  depth line from known parameters, checks the identity closes to 1e-13.
- `test_log_qc_matches_the_bracketed_term` — checks `logQc` against the
  bracketed product written out directly.
- `test_jacobian_term_is_present_and_has_the_right_sign` — guards the change
  of variables, which is silent if lost.
- `test_z0_prior_reparameterisation_is_equivalent` — checks the truncated
  normal on `depth_min` equals the normal on `z0`.

## Output

```
{reach_id}_geobam_wse.nc
  /r /logn /logWb /logDb /f /logDc /z0   mean (nx), sd (scalar)
  /q                                     q (nt), q_sd (nt)
```
