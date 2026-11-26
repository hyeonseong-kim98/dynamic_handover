"""Command-line options shared by the robot trajectory estimators."""

import argparse
from pathlib import Path

from SparseIAAL.dataset import resolve_motion_directory


REPOSITORY_DIR = Path(__file__).resolve().parents[1]


def parse_estimator_args(args=None, render_default=True):
    parser = argparse.ArgumentParser(description="Dynamic handover trajectory estimator")
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=REPOSITORY_DIR / "run_ws" / "data" / "dataset",
        help="Public dataset checkout or directory containing motion_*.pkl files",
    )
    parser.add_argument(
        "--motion-type", choices=("normal", "variation", "all"), default="normal",
        help="Motion group to load (default: normal; variation is reserved for OOD evaluation)",
    )
    parser.add_argument("--max-data-num", type=int, default=None)
    parser.add_argument(
        "--kernel-cache", type=Path,
        default=REPOSITORY_DIR / ".cache" / "spgp_kernel.pkl",
    )
    parser.add_argument("--no-kernel-cache", action="store_true")
    parser.add_argument("--render", dest="render", action="store_true")
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.set_defaults(render=render_default)
    options = parser.parse_args(args)
    if options.max_data_num is not None and options.max_data_num < 1:
        parser.error("--max-data-num must be positive")
    try:
        options.motion_dir = resolve_motion_directory(options.dataset_dir)
    except FileNotFoundError as error:
        parser.error(str(error))
    options.kernel_cache = (
        None if options.no_kernel_cache else options.kernel_cache.expanduser().resolve()
    )
    return options
