"""Locate motions in the public dataset or an existing motion directory."""

from pathlib import Path
import re


MOTION_PATTERN = re.compile(r"motion_(normal|variation)_(\d+)\.pkl$")


def resolve_motion_directory(dataset_dir):
    path = Path(dataset_dir).expanduser().resolve()
    for candidate in (path / "data" / "whole_motions", path / "whole_motions", path):
        if candidate.is_dir() and any(candidate.glob("motion_*.pkl")):
            return candidate
    raise FileNotFoundError(
        "No motion files found in {}. Run git submodule update --init --recursive "
        "from the repository root, or pass a dataset checkout with --dataset-dir.".format(path)
    )


def list_motion_files(directory, motion_type="all", max_data_num=None):
    if motion_type not in ("normal", "variation", "all"):
        raise ValueError("motion_type must be normal, variation, or all")
    if max_data_num is not None and max_data_num < 1:
        raise ValueError("max_data_num must be positive")
    files = []
    for path in Path(directory).iterdir():
        match = MOTION_PATTERN.fullmatch(path.name)
        if path.is_file() and match and (motion_type == "all" or match[1] == motion_type):
            files.append((match[1], int(match[2]), path.name))
    filenames = [entry[2] for entry in sorted(files)]
    if max_data_num is not None:
        filenames = filenames[:max_data_num]
    if not filenames:
        raise ValueError("No {} motion files found in {}".format(motion_type, directory))
    return filenames
