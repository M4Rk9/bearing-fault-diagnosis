"""M4 manifest-driven ingestion. Run with python -m src.data_pipeline --help."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import resample_poly

CLASSES = {"normal": 0, "inner_race": 1, "ball": 2, "outer_race": 3}
PARTITIONS = ("train", "validation", "test")
REQUIRED = {
    "record_id", "group_id", "relative_path", "sha256", "source_url",
    "class_name", "fault_diameter_in", "fault_depth_in", "manufacturer",
    "load_hp", "rpm", "rpm_source", "faulted_bearing_end", "channel_key",
    "sensor_end", "sampling_rate_hz", "sampling_rate_source",
    "outer_race_position", "units", "calibration_source", "partition",
}


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def positive_integer(value):
    return type(value) is int and value > 0


def validate_manifest(rows, require_hash=True, require_coverage=True):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON list")
    identities, groups, hashes, paths = set(), {}, {}, {}
    channels = set()
    coverage = {p: set() for p in PARTITIONS}
    for row in rows:
        if not isinstance(row, dict) or REQUIRED - row.keys():
            raise ValueError("Manifest row missing required fields")
        for key in ("record_id", "group_id", "relative_path", "source_url", "channel_key"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Invalid {key}")
        identity = row["record_id"]
        channel_identity = (row["group_id"], row["channel_key"])
        if channel_identity in channels:
            raise ValueError("Duplicate channel within recording group")
        channels.add(channel_identity)
        if identity in identities:
            raise ValueError("Duplicate record_id")
        identities.add(identity)
        path = Path(row["relative_path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must stay within raw directory")
        if row["class_name"] not in CLASSES or row["partition"] not in PARTITIONS:
            raise ValueError("Invalid class or partition")
        if row["sensor_end"] not in ("DE", "FE", "BA") or not row["channel_key"].endswith("_" + row["sensor_end"] + "_time"):
            raise ValueError("Channel key and sensor_end disagree")
        if not positive_integer(row["sampling_rate_hz"]) or not row["sampling_rate_source"]:
            raise ValueError("Verified per-channel sampling rate and source required")
        for key in ("load_hp", "rpm"):
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid {key}")
        if row["rpm"] <= 0 or not row["rpm_source"]:
            raise ValueError("RPM and source required")
        normal = row["class_name"] == "normal"
        if normal:
            if row["faulted_bearing_end"] is not None or row["fault_diameter_in"] is not None:
                raise ValueError("Healthy record must have null fault metadata")
        elif row["faulted_bearing_end"] not in ("DE", "FE"):
            raise ValueError("Faulted bearing end required")
        for key in ("fault_diameter_in", "fault_depth_in"):
            value = row[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"Invalid {key}")
        if not normal and row["fault_diameter_in"] is None:
            raise ValueError("Fault diameter required")
        if row["class_name"] == "outer_race":
            if row["outer_race_position"] not in ("3:00", "6:00", "12:00"):
                raise ValueError("Outer-race position required")
        elif row["outer_race_position"] is not None:
            raise ValueError("Position only applies to outer-race faults")
        if row["units"] != "unknown" and not row["calibration_source"]:
            raise ValueError("Known units require calibration source")
        sha = row["sha256"]
        if sha is not None and (not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("Invalid SHA-256")
        if require_hash and sha is None:
            raise ValueError("Raw-file SHA-256 required; run acquire first")
        # Same original recording may have multiple channels, but never different labels/loads/partitions.
        signature = tuple(row[k] for k in ("partition", "class_name", "load_hp", "rpm", "fault_diameter_in", "fault_depth_in", "manufacturer", "faulted_bearing_end", "outer_race_position"))
        group = row["group_id"]
        if group in groups and groups[group] != signature:
            raise ValueError("Recording group has inconsistent metadata or crosses partitions")
        groups[group] = signature
        for mapping, key in ((hashes, sha), (paths, str(path))):
            if key is not None and key in mapping and mapping[key] != group:
                raise ValueError("Duplicate file assigned to different recording groups")
            if key is not None:
                mapping[key] = group
        coverage[row["partition"]].add(row["class_name"])
    if require_coverage and any(classes != set(CLASSES) for classes in coverage.values()):
        raise ValueError("Every partition must contain all four classes")


def raw_path(root, row):
    root = Path(root).resolve()
    path = (root / row["relative_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Raw path escapes root")
    return path


def read_signal(path, channel_key):
    mat = loadmat(path)
    if channel_key not in mat:
        raise ValueError(f"Missing requested channel {channel_key}; available: {sorted(k for k in mat if not k.startswith('__'))}")
    value = np.asarray(mat[channel_key])
    if value.dtype.kind not in "iuf" or value.ndim not in (1, 2) or (value.ndim == 2 and 1 not in value.shape):
        raise ValueError("Channel must be a real numeric vector")
    signal = value.reshape(-1).astype(np.float64)
    if len(signal) < 2 or not np.isfinite(signal).all() or np.ptp(signal) == 0:
        raise ValueError("Signal is empty, short, nonfinite or constant")
    return signal


def make_windows(signal, source_rate, target_rate=12000, window_size=2048, hop_size=1024):
    if not all(positive_integer(v) for v in (source_rate, target_rate, window_size, hop_size)) or hop_size > window_size:
        raise ValueError("Rates, window and hop must be positive integers; hop <= window")
    signal = np.asarray(signal)
    if signal.ndim != 1 or signal.dtype.kind not in "iuf" or not np.isfinite(signal).all():
        raise ValueError("Expected finite real 1D signal")
    if len(signal) < 2 or np.ptp(signal) == 0:
        raise ValueError("Short or constant signal")
    signal = signal.astype(np.float64)
    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        signal = resample_poly(signal, target_rate // divisor, source_rate // divisor,
                               window=("kaiser", 5.0), padtype="constant")
    if len(signal) < window_size:
        raise ValueError("Signal shorter than one target-rate window")
    starts = np.arange(0, len(signal) - window_size + 1, hop_size)
    raw = np.stack([signal[s:s + window_size] for s in starts])
    features = raw - raw.mean(axis=1, keepdims=True)
    scales = features.std(axis=1, keepdims=True)
    if np.any(scales <= 1e-12):
        raise ValueError("Constant or near-constant window")
    cnn = features / np.maximum(scales, 1e-8)
    return features, cnn, starts, len(signal)


def acquire(rows, root):
    """Download only the official CWRU host; preserve existing files and pin hashes."""
    from urllib.parse import urlparse
    result = []
    for row in rows:
        row = dict(row)
        url = urlparse(row["source_url"])
        if url.scheme != "https" or url.hostname != "engineering.case.edu" or not url.path.endswith(".mat"):
            raise ValueError("Downloads require an official HTTPS CWRU .mat URL")
        path = raw_path(root, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temp = path.with_suffix(".mat.partial")
            try:
                with urlopen(row["source_url"], timeout=30) as response, open(temp, "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                read_signal(temp, row["channel_key"])
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        read_signal(path, row["channel_key"])
        actual = digest(path)
        if row["sha256"] is not None and row["sha256"] != actual:
            raise ValueError(f"Hash mismatch: {path}")
        row["sha256"] = actual
        result.append(row)
    return result


def prepare(rows, root, output, target_rate=12000, window_size=2048, hop_size=1024, require_coverage=True):
    validate_manifest(rows, require_coverage=require_coverage)
    by_partition = {p: [] for p in PARTITIONS}
    records, fingerprints = [], {}
    for row in rows:
        path = raw_path(root, row)
        if digest(path) != row["sha256"]:
            raise ValueError(f"Hash mismatch: {path}")
        signal = read_signal(path, row["channel_key"])
        fingerprint = hashlib.sha256(signal.astype("<f8").tobytes()).hexdigest()
        if fingerprint in fingerprints and fingerprints[fingerprint] != row["group_id"]:
            raise ValueError("Identical channel content in different recording groups")
        fingerprints[fingerprint] = row["group_id"]
        features, cnn, starts, length = make_windows(signal, row["sampling_rate_hz"], target_rate, window_size, hop_size)
        by_partition[row["partition"]].append((row, features, cnn, starts))
        mat = loadmat(path)
        rpm_key = row["channel_key"].split("_")[0] + "RPM"
        rpm_value = mat.get(rpm_key)
        measured_rpm = None
        if rpm_value is not None:
            rpm_array = np.asarray(rpm_value)
            if rpm_array.size != 1 or rpm_array.dtype.kind not in "iuf" or not np.isfinite(rpm_array).all() or float(rpm_array.item()) <= 0:
                raise ValueError(f"Invalid measured RPM in {rpm_key}")
            measured_rpm = float(rpm_array.item())
        records.append({**row, "measured_rpm": measured_rpm, "measured_rpm_key": rpm_key if measured_rpm is not None else None, "signal_sha256": fingerprint, "source_samples": len(signal), "target_samples": length,
                        "windows": len(starts), "discarded_tail_samples": length - int(starts[-1]) - window_size})
    # Validate everything before writing any output.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid mixing experiments")
    for partition, items in by_partition.items():
        if not items:
            continue
        np.savez_compressed(output / f"{partition}.npz",
            feature_windows=np.concatenate([i[1] for i in items]),
            cnn_windows=np.concatenate([i[2] for i in items]),
            labels=np.concatenate([np.full(len(i[3]), CLASSES[i[0]["class_name"]]) for i in items]),
            record_ids=np.concatenate([np.full(len(i[3]), i[0]["record_id"]) for i in items]),
            group_ids=np.concatenate([np.full(len(i[3]), i[0]["group_id"]) for i in items]),
            start_samples=np.concatenate([i[3] for i in items]))
    manifest_bytes = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    summary = {"schema_version": "m4-v1", "classification_ready": require_coverage,
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "classes": CLASSES,
               "pipeline_sha256": digest(Path(__file__)),
               "target_rate_hz": target_rate, "window_size": window_size, "hop_size": hop_size,
               "window_duration_s": window_size / target_rate,
               "resampling": "scipy.signal.resample_poly; Kaiser beta=5; constant zero padding",
               "feature_preprocessing": "per-window mean removal; amplitude retained; no bandpass",
               "cnn_preprocessing": "feature branch / max(population std, 1e-8); reject std <= 1e-12",
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
               "records": records,
               "partitions": {p: {"groups": sorted({i[0]["group_id"] for i in items}),
                                  "windows": sum(len(i[3]) for i in items),
                                  "by_class": {name: {"groups": sorted({i[0]["group_id"] for i in items if i[0]["class_name"] == name}),
                                                       "windows": sum(len(i[3]) for i in items if i[0]["class_name"] == name)} for name in CLASSES}}
                              for p, items in by_partition.items()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "prepare"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", required=True)
    parser.add_argument("--record-example", action="store_true", help="Allow incomplete class coverage; never classification evidence")
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text())
    if args.command == "acquire":
        # Acquisition can preserve unresolved sampling metadata; preparation cannot.
        locked = acquire(rows, args.raw_dir)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            json.dump(locked, stream, indent=2)
    else:
        prepare(rows, args.raw_dir, args.output, require_coverage=not args.record_example)


if __name__ == "__main__":
    main()
