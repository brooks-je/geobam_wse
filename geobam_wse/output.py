"""Write geobam_wse output to NetCDF.

Port of output.R. The file layout is unchanged from the R module so that
downstream Confluence stages (moi, offline, validation) need no adjustment:

    /r, /logn, /logWb, /logDb, /dz, /z0   each with mean (nx) and sd (scalar)
    /q                                    with q (nt) and q_sd (nt)
"""

import numpy as np

FILL = -999999999999.0

POSTERIOR_GROUPS = ("r", "logn", "logWb", "logDb", "f", "logDc", "z0")


def concatenate_invalid(values, invalid_times):
    """Reinsert NaN at the timesteps that were dropped before sampling."""
    out = list(np.asarray(values, dtype=float))
    for index in sorted(np.asarray(invalid_times, dtype=int)):
        out.insert(int(index), np.nan)
    return np.asarray(out, dtype=float)


def _fill_nan(values):
    v = np.asarray(values, dtype=float)
    return np.where(np.isfinite(v), v, FILL)


def concatenate_invalid_nodes(values, invalid_nodes, nx):
    """Reinsert NaN at the nodes that were dropped before sampling.

    output.R declared the nx dimension from the full SoS node list but wrote
    posteriors of the post-screening length, so any dropped node produced a
    dimension mismatch. Nodes get the same pad-back treatment as timesteps.
    """
    out = list(np.asarray(values, dtype=float))
    for index in sorted(np.asarray(invalid_nodes, dtype=int)):
        if index <= len(out):
            out.insert(int(index), np.nan)
    out = np.asarray(out, dtype=float)
    if out.size < nx:
        out = np.concatenate([out, np.full(nx - out.size, np.nan)])
    return out[:nx]


def _write_posteriors(nc_out, posteriors, is_valid, nx, invalid_nodes):
    for name in POSTERIOR_GROUPS:
        grp = nc_out.createGroup(name)

        var_mean = grp.createVariable("mean", "f8", ("nx",), fill_value=FILL)
        var_sd = grp.createVariable("sd", "f8", fill_value=FILL)

        if is_valid and name in posteriors:
            mean = np.asarray(posteriors[name]["mean"], dtype=float).ravel()
            mean = concatenate_invalid_nodes(mean, invalid_nodes, nx)
            var_mean[:] = _fill_nan(mean)
            var_sd[...] = float(posteriors[name]["sd"])
        else:
            var_mean[:] = np.full(nx, FILL)
            var_sd[...] = FILL


def _write_discharge(nc_out, discharge, discharge_sd, is_valid, nt):
    grp = nc_out.createGroup("q")

    var_q = grp.createVariable("q", "f8", ("nt",), fill_value=FILL)
    var_sd = grp.createVariable("q_sd", "f8", ("nt",), fill_value=FILL)

    if is_valid:
        var_q[:] = _fill_nan(np.asarray(discharge, dtype=float).ravel())
        var_sd[:] = _fill_nan(np.asarray(discharge_sd, dtype=float).ravel())
    else:
        var_q[:] = np.full(nt, FILL)
        var_sd[:] = np.full(nt, FILL)


def write_output(data, posteriors, discharge, discharge_sd, out_dir, is_valid,
                 obs_times):
    """Write one reach to {out_dir}/{reach_id}_geobam_wse.nc."""
    from netCDF4 import Dataset   # imported here so the pure
                                  # helpers above stay importable
                                  # without a NetCDF stack

    print("writing output...", flush=True)

    nt = int(np.asarray(data["nt"]).size)
    node_ids = np.asarray(data["node_ids"]).ravel()
    nx = int(node_ids.size)

    if is_valid:
        discharge = concatenate_invalid(discharge, data["invalid_times"])
        discharge_sd = concatenate_invalid(discharge_sd, data["invalid_times"])

    nc_file = f"{out_dir}/{data['reach_id']}_geobam_wse.nc"

    with Dataset(nc_file, "w", format="NETCDF4") as nc_out:
        nc_out.reach_id = np.int64(data["reach_id"])
        nc_out.node_ids = np.asarray(node_ids, dtype=np.int64)

        nc_out.createDimension("nt", nt)
        var_nt = nc_out.createVariable("nt", "i4", ("nt",))
        var_nt.units = "time"
        var_nt[:] = np.arange(nt, dtype=np.int32)

        var_time_str = nc_out.createVariable("time_str", str, ("nt",))
        var_time_str.units = "time"
        times = np.asarray(obs_times, dtype=object).ravel()
        for i in range(nt):
            if i < times.size:
                value = times[i]
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                var_time_str[i] = str(value).strip()
            else:
                var_time_str[i] = ""

        nc_out.createDimension("nx", nx)
        var_nx = nc_out.createVariable("nx", "i4", ("nx",))
        var_nx.units = "num_nodes"
        var_nx[:] = np.arange(nx, dtype=np.int32)

        print("writing posteriors...", flush=True)
        _write_posteriors(nc_out, posteriors, is_valid, nx,
                          data.get("invalid_nodes", []))

        print("writing discharge...", flush=True)
        _write_discharge(nc_out, discharge, discharge_sd, is_valid, nt)

    return nc_file
