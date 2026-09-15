"""Flow-law arithmetic that needs nothing but numpy.

Kept separate from engine.py so the ragged index mapping and the discharge
reconstruction can be imported and tested without pulling in JAX.
"""

import numpy as np


def build_index_arrays(hasdat):
    """Flatten the nx x nt mask into per-observation node and time indices.

    Returns (xind, tind), each length ntot, in row-major order so they line up
    with a hasdat-masked flatten of any nx x nt matrix.
    """
    xind, tind = np.nonzero(np.asarray(hasdat) == 1)
    return xind.astype(np.int32), tind.astype(np.int32)


def log_qc(r, logWb, logDb, logn, logDc, logS):
    """Centre-point discharge, on the log scale.

    logQc = logWb - (1/r) logDb + (1/r)(log(r+1) - log r)
            + (1/2) logS - logn + (5/3 + 1/r) logDc

    All arguments broadcast together, so this works per node (with a scalar
    logS) or per observation (with everything already expanded).
    """
    invr = 1.0 / r
    return (logWb
            - invr * logDb
            + invr * (np.log(r + 1.0) - np.log(r))
            + 0.5 * logS
            - logn
            + (5.0 / 3.0 + invr) * logDc)


def remake_discharge(Sobs, Hobs, posteriors, hasdat=None):
    """Rebuild discharge from posterior means (diagnostic, not the reported Q).

    Inverts the depth line: logQ = logQc + (1/f)(log d - logDc).

    Width is not in the flow law, so it gates nothing here.
    """
    r = np.asarray(posteriors["r"]["mean"], dtype=float)
    logn = np.asarray(posteriors["logn"]["mean"], dtype=float)
    z0 = np.asarray(posteriors["z0"]["mean"], dtype=float)
    logWb = np.asarray(posteriors["logWb"]["mean"], dtype=float)
    logDb = np.asarray(posteriors["logDb"]["mean"], dtype=float)
    f = np.asarray(posteriors["f"]["mean"], dtype=float)
    logDc = np.asarray(posteriors["logDc"]["mean"], dtype=float)

    Sobs = np.asarray(Sobs, dtype=float)
    Hobs = np.asarray(Hobs, dtype=float)

    if hasdat is None:
        hasdat = np.isfinite(Sobs) & (Sobs > 0) & np.isfinite(Hobs)
    hasdat = np.asarray(hasdat).astype(bool)

    nx, nt = Sobs.shape
    finalQ = np.full((nx, nt), np.nan)

    for i in range(nx):
        # Missing cells arrive as placeholder zeros. Under a width formulation
        # log(0) = -inf was caught downstream; with depth, H = 0 gives a
        # finite but meaningless number, so mask explicitly.
        d = Hobs[i] - z0[i]
        d = np.where(hasdat[i] & np.isfinite(d) & (d > 0), d, np.nan)

        S = np.where(hasdat[i] & np.isfinite(Sobs[i]) & (Sobs[i] > 0),
                     Sobs[i], np.nan)

        with np.errstate(invalid="ignore", divide="ignore"):
            logQc = log_qc(r[i], logWb[i], logDb[i], logn[i], logDc[i],
                           np.log(S))
            logQ = (1.0 / f[i]) * (np.log(d) - logDc[i]) + logQc
            finalQ[i] = np.exp(logQ)

    finalQ[~np.isfinite(finalQ)] = np.nan
    with np.errstate(invalid="ignore"):
        out = np.nanmean(finalQ, axis=0)
    return out
