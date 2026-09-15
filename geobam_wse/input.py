"""Input operations for geobam_wse.

Port of input.R. Reads SWOT observations and SoS priors, screens them, and
drops nodes and timesteps that carry too little data.

SOS:  monthly_q, max_q, min_q, logQ_sd
SWOT: node/time, node/width, node/slope2, node/wse
"""

import random
import time
import warnings

import numpy as np
import pandas as pd
from netCDF4 import Dataset, chartostring

# SWOT fill magnitude. Negative WSE is physically legitimate (below geoid), so
# wse is screened on magnitude rather than on sign.
WSE_MIN = -1000.0
WSE_MAX = 9000.0

# A node or timestep is dropped when it has fewer than this many valid cells.
MIN_VALID_MARGIN = 3


def _to_nx_nt(arr, nx, nt):
    """Return arr oriented (nx, nt).

    RNetCDF reverses dimension order and input.R compensated with t(). The
    Python netCDF4 reader already returns declaration order, so usually no
    transpose is needed -- but check rather than assume, because a square
    nx == nt case would hide a mistake forever.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D variable, got shape {a.shape}")
    if a.shape == (nx, nt):
        return a
    if a.shape == (nt, nx):
        return a.T
    raise ValueError(
        f"variable shape {a.shape} matches neither (nx, nt) = ({nx}, {nt}) "
        f"nor its transpose")


def read_strings(var):
    """Read a NetCDF string variable as a 1-D array of Python str.

    NetCDF stores text two ways and the reader treats them very differently:

      NC_STRING  -> object array of str, shape (nt,)
      NC_CHAR    -> array of single characters, shape (nt, strlen)

    Flattening the second one gives nt * strlen individual letters rather than
    nt timestamps, so the character case has to be joined back up first.
    RNetCDF did this implicitly, which is why input.R never needed it.
    """
    raw = var[:]
    if isinstance(raw, np.ma.MaskedArray):
        raw = raw.filled(b"" if raw.dtype.kind == "S" else "")

    arr = np.asarray(raw)

    if arr.dtype.kind in ("S", "a") and arr.ndim >= 2:
        arr = np.asarray(chartostring(arr))

    arr = arr.ravel()

    out = []
    for item in arr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", errors="replace").strip())
        else:
            out.append(str(item).strip())
    return np.array(out, dtype=object)


def _masked_to_nan(arr):
    """netCDF4 hands back masked arrays; make the mask explicit NaN."""
    if isinstance(arr, np.ma.MaskedArray):
        return arr.filled(np.nan).astype(float)
    return np.asarray(arr, dtype=float)


def get_swot(swot_file):
    """Read the SWOT observation file."""
    with Dataset(swot_file, "r") as ds:
        nx = np.asarray(ds.variables["nx"][:]).ravel()
        nt = np.asarray(ds.variables["nt"][:]).ravel()
        n_x, n_t = nx.size, nt.size

        node = ds.groups["node"]
        width = _to_nx_nt(_masked_to_nan(node.variables["width"][:]), n_x, n_t)
        slope2 = _to_nx_nt(_masked_to_nan(node.variables["slope2"][:]), n_x, n_t)
        wse = _to_nx_nt(_masked_to_nan(node.variables["wse"][:]), n_x, n_t)
        t_sec = _to_nx_nt(_masked_to_nan(node.variables["time"][:]), n_x, n_t)

        obs_times = read_strings(ds.groups["reach"].variables["time_str"])
        if obs_times.size != n_t:
            warnings.warn(
                f"time_str has {obs_times.size} entries but nt is {n_t}; "
                f"the output time_str will be padded or truncated to match")

    return {
        "nx": nx, "nt": nt,
        "width": width, "slope2": slope2, "wse": wse, "time": t_sec,
        "obs_times": obs_times,
    }


def get_sos(sos_file, reach_id, retries=5):
    """Read the SoS priors for one reach.

    input.R retried on failure but decremented a local copy of the counter
    inside the handler closure, so the loop could never terminate. This one
    decrements properly and gives up.
    """
    last_error = None
    for attempt in range(retries):
        try:
            return _read_sos(sos_file, reach_id)
        except Exception as exc:                        # noqa: BLE001
            last_error = exc
            if attempt == retries - 1:
                break
            delay = random.uniform(1.0, 10.0)
            warnings.warn(
                f"SoS read failed ({exc}); retry {attempt + 1}/{retries - 1} "
                f"in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"could not read {sos_file} for reach {reach_id}: "
                       f"{last_error}")


def _read_sos(sos_file, reach_id):
    q_priors = {}
    with Dataset(sos_file, "r") as ds:
        rids = np.asarray(ds.groups["reaches"].variables["reach_id"][:]).ravel()
        idx = np.nonzero(rids == reach_id)[0]
        if idx.size == 0:
            raise KeyError(f"reach {reach_id} absent from the SoS")
        idx = int(idx[0])

        nodes = ds.groups["nodes"]
        nrids = np.asarray(nodes.variables["reach_id"][:]).ravel()
        node_idx = np.nonzero(nrids == reach_id)[0]
        nids = np.asarray(nodes.variables["node_id"][:]).ravel()
        q_priors["nids"] = nids[node_idx]

        model = ds.groups["model"]
        monthly = _masked_to_nan(model.variables["monthly_q"][:])
        monthly = np.asarray(monthly)
        # monthly_q is (12, nreaches) or its transpose
        if monthly.ndim == 2 and monthly.shape[0] != 12:
            monthly = monthly.T
        with np.errstate(divide="ignore", invalid="ignore"):
            q_priors["logQ_hat_monthly"] = np.log(monthly[:, idx])

        max_q = float(_masked_to_nan(model.variables["max_q"][:]).ravel()[idx])
        min_q = float(_masked_to_nan(model.variables["min_q"][:]).ravel()[idx])

        with np.errstate(divide="ignore", invalid="ignore"):
            q_priors["upperbound_logQ"] = np.log(max_q)
            if not np.isfinite(min_q):
                q_priors["lowerbound_logQ"] = np.nan
            else:
                if min_q == 0:
                    min_q = 0.01
                q_priors["lowerbound_logQ"] = np.log(min_q)

        gb_reach = ds.groups["gbpriors"].groups["reach"]
        q_priors["logQ_sd"] = float(
            _masked_to_nan(gb_reach.variables["logQ_sd"][:]).ravel()[idx])

    return {"Q_priors": q_priors}


def get_invalid(obs):
    """Nodes and timesteps with too few valid cells.

    Matches the R thresholds: a node is invalid when its count of missing
    cells is at least (nt - 3), and likewise for timesteps across nodes.
    """
    missing = ~np.isfinite(obs)
    nrow, ncol = obs.shape
    invalid_nodes = missing.sum(axis=1) >= (ncol - MIN_VALID_MARGIN)
    invalid_times = missing.sum(axis=0) >= (nrow - MIN_VALID_MARGIN)
    return invalid_nodes, invalid_times


def get_invalid_nodes_times(slope2, t_sec, wse, width=None):
    """Union of the invalid nodes and timesteps across the screened variables.

    width is optional. It does not appear in the flow law, but classify() in
    priors.py needs it to build the dz prior, so by default it is still
    considered here. Pass width=None to stop it gating anything.
    """
    node_masks, time_masks = [], []
    for arr in (slope2, t_sec, wse, width):
        if arr is None:
            continue
        n, t = get_invalid(arr)
        node_masks.append(n)
        time_masks.append(t)

    invalid_nodes = np.nonzero(np.any(node_masks, axis=0))[0]
    invalid_times = np.nonzero(np.any(time_masks, axis=0))[0]
    return invalid_nodes, invalid_times


def _monthly_to_timestep(logQ_hat_monthly, t_sec):
    """Expand the 12-month prior onto the observation timesteps.

    time is nx x nt seconds since 2000-01-01; only an nt vector is wanted.
    Months with no SoS value fall back to the mean of those that have one.
    """
    days = np.asarray(t_sec)[0, :] / 86400.0
    dates = pd.to_datetime("2000-01-01") + pd.to_timedelta(days, unit="D")
    months = pd.DataFrame({"month": dates.month})

    table = pd.DataFrame({
        "month": np.arange(1, 13),
        "logQ_hat": np.asarray(logQ_hat_monthly, dtype=float).ravel(),
    })

    merged = months.merge(table, on="month", how="left")
    values = merged["logQ_hat"].to_numpy(dtype=float)
    if np.isnan(values).any():
        fill = np.nanmean(values) if np.isfinite(values).any() else np.nan
        values = np.where(np.isnan(values), fill, values)
    return values


def check_observations(swot_data, sos_data, width_gates_validity=True):
    """Screen the observations and drop invalid nodes and timesteps.

    Returns None when nothing usable survives.
    """
    q = sos_data["Q_priors"]
    if (not np.isfinite(q["logQ_hat_monthly"]).any()
            or not np.isfinite(q["upperbound_logQ"])
            or not np.isfinite(q["lowerbound_logQ"])
            or not np.isfinite(q["logQ_sd"])):
        return None

    width = swot_data["width"].copy()
    slope2 = swot_data["slope2"].copy()
    wse = swot_data["wse"].copy()
    t_sec = swot_data["time"].copy()

    width[~np.isfinite(width) | (width < 0)] = np.nan
    slope2[~np.isfinite(slope2) | (slope2 < 0)] = np.nan
    wse[~np.isfinite(wse) | (wse < WSE_MIN) | (wse > WSE_MAX)] = np.nan

    logQ_hat = _monthly_to_timestep(q["logQ_hat_monthly"], t_sec)

    invalid_nodes, invalid_times = get_invalid_nodes_times(
        slope2, t_sec, wse, width if width_gates_validity else None)

    keep_nodes = np.setdiff1d(np.arange(width.shape[0]), invalid_nodes)
    keep_times = np.setdiff1d(np.arange(width.shape[1]), invalid_times)

    def cut(a):
        return a[np.ix_(keep_nodes, keep_times)]

    width, slope2, wse, t_sec = cut(width), cut(slope2), cut(wse), cut(t_sec)
    logQ_hat = logQ_hat[keep_times]

    for name, arr in (("slope2", slope2), ("wse", wse), ("time", t_sec)):
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
            print(f"  {name} reduced to {arr.shape}; too little data")
            return None

    q = dict(q)
    q["logQ_hat"] = logQ_hat

    return {
        "swot_data": {"width": width, "slope2": slope2, "wse": wse,
                      "time": t_sec},
        "sos_data": {"Q_priors": q},
        "invalid_nodes": invalid_nodes,
        "invalid_times": invalid_times,
    }


def get_input(swot_file, sos_file, reach_id, width_gates_validity=True):
    """Assemble everything the model needs for one reach."""
    swot_data = get_swot(swot_file)
    sos_data = get_sos(sos_file, reach_id)

    data = check_observations(swot_data, sos_data, width_gates_validity)

    if data is None:
        print("in get_input(), geobam_wse has decided the data are invalid")
        return {
            "valid": False,
            "reach_id": reach_id,
            "obs_times": swot_data["obs_times"],
            "nx": swot_data["nx"],
            "nt": swot_data["nt"],
            "node_ids": sos_data["Q_priors"]["nids"],
        }

    return {
        "valid": True,
        "reach_id": reach_id,
        "obs_times": swot_data["obs_times"],
        "nt": swot_data["nt"],
        "swot_data": data["swot_data"],
        "sos_data": data["sos_data"],
        "invalid_nodes": data["invalid_nodes"],
        "invalid_times": data["invalid_times"],
        "node_ids": sos_data["Q_priors"]["nids"],
    }
