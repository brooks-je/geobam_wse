#!/usr/bin/env python3
"""Run geobam_wse on one reach and write the result to the output mount.

Port of run_geobam_wse.R.

    python run_geobam_wse.py -r reaches.json -i 0

The index is 0-based, matching AWS Batch. When -i is omitted (or given as
-256) it is taken from AWS_BATCH_JOB_ARRAY_INDEX instead.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# JAX fixes its device count the first time it is touched, and NumPyro runs
# one chain per host device. This has to happen before anything imports jax,
# which is why it sits above the module imports rather than in engine.py.
_CHAINS = int(float(os.environ.get("GEOBAM_WSE_CHAINS") or 3))
os.environ.setdefault(
    "XLA_FLAGS", f"--xla_force_host_platform_device_count={_CHAINS}")

import numpyro                                    # noqa: E402

numpyro.set_host_device_count(_CHAINS)

from geobam_wse.input import get_input           # noqa: E402
from geobam_wse.output import write_output        # noqa: E402
from geobam_wse.process import process_data       # noqa: E402

IN_DIR = "/mnt/data/input"
OUT_DIR = "/mnt/data/output"
TMP_PATH = "/tmp"

ENV_INDEX = -256


def get_reach_files(reaches_json, index, bucket_key):
    """Resolve the SWOT and SoS paths for one entry of reaches.json."""
    with open(os.path.join(IN_DIR, reaches_json)) as fh:
        entry = json.load(fh)[index]

    if bucket_key:
        # Deferred so a local run needs neither boto3 nor credentials.
        sys.path.insert(0, "/app/sos_read")
        from sos_read import download_sos          # noqa: PLC0415

        sos_filepath = os.path.join(TMP_PATH, entry["sos"])
        download_sos(bucket_key, sos_filepath)
    else:
        sos_filepath = os.path.join(IN_DIR, "sos", entry["sos"])

    return {
        "reach_id": entry["reach_id"],
        "swot_file": os.path.join(IN_DIR, "swot", entry["swot"]),
        "sos_file": sos_filepath,
    }


def create_invalid_out(nt):
    """Placeholder result for a reach with nothing usable in it."""
    nt_vector = np.full(nt, np.nan)
    base = {"mean": np.nan, "sd": np.nan}
    return {
        "posterior_Q": nt_vector,
        "posterior_Q_sd": nt_vector,
        "posteriors": {k: dict(base) for k in
                       ("r", "logn", "logWb", "logDb", "f", "logDc", "z0")},
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run geobam_wse on one reach")
    parser.add_argument("-i", "--index", type=int, default=ENV_INDEX,
                        help="0-based index into reaches.json")
    parser.add_argument("-b", "--bucket_key", type=str, default="",
                        help="Bucket key to find the SoS")
    parser.add_argument("-r", "--reaches_json", type=str,
                        default="reaches.json", help="Name of reaches.json")
    parser.add_argument("--seed", type=int, default=0,
                        help="PRNG seed for the sampler")
    parser.add_argument("--width-gates-hasdat", action="store_true",
                        help="Require width > 0 for a cell to enter the "
                             "likelihood, as process.R did. Width is not in "
                             "the flow law; use this only to reproduce the R "
                             "module's cell selection when diffing.")
    return parser.parse_args(argv)


def main(argv=None):
    start = time.time()
    opts = parse_args(argv)

    index = opts.index
    if index is None or index == ENV_INDEX:
        index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 0))

    print(f"bucket_key: {opts.bucket_key}")
    print(f"index: {index}")
    print(f"reaches_json: {opts.reaches_json}")

    # Fail on an unwritable output mount now rather than after an hour of
    # sampling. This is a permissions problem often enough to be worth it.
    os.makedirs(OUT_DIR, exist_ok=True)
    probe = os.path.join(OUT_DIR, ".geobam_wse_write_probe")
    try:
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise SystemExit(f"output directory {OUT_DIR} is not writable: {exc}")

    io_data = get_reach_files(opts.reaches_json, index, opts.bucket_key)
    print(f"reach_id: {io_data['reach_id']}")
    print(f"swot_file: {io_data['swot_file']}")
    print(f"sos_file: {io_data['sos_file']}")

    in_data = get_input(io_data["swot_file"], io_data["sos_file"],
                        io_data["reach_id"])

    print("processing data...", flush=True)
    if in_data["valid"]:
        print("data was valid...", flush=True)
        result = process_data(in_data, seed=opts.seed,
                              width_gates_hasdat=opts.width_gates_hasdat)
        out_data = {
            "reach_id": io_data["reach_id"],
            "nt": in_data["nt"],
            "invalid_nodes": in_data["invalid_nodes"],
            "invalid_times": in_data["invalid_times"],
            "node_ids": in_data["node_ids"],
        }
    else:
        nt = int(np.asarray(in_data["nt"]).size)
        result = create_invalid_out(nt)
        out_data = {
            "reach_id": io_data["reach_id"],
            "nt": in_data["nt"],
            "invalid_nodes": [],
            "invalid_times": [],
            "node_ids": in_data["node_ids"],
        }

    nc_file = write_output(
        data=out_data,
        posteriors=result["posteriors"],
        discharge=result["posterior_Q"],
        discharge_sd=result["posterior_Q_sd"],
        out_dir=OUT_DIR,
        is_valid=in_data["valid"],
        obs_times=in_data["obs_times"],
    )

    print(f"wrote {nc_file}")
    print(f"Total execution time for reach {io_data['reach_id']}: "
          f"{time.time() - start:.1f} seconds.")


if __name__ == "__main__":
    main()
