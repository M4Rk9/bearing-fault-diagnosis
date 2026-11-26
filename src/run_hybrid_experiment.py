"""Run the hybrid experiment directly from the supplied numerical ZIP archive."""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from .train_hybrid import DRIVE_CLASSES, STORAGE_SPLITS, VARIANTS, RAW_DIR
from .report_hybrid import generate


def extract_needed(archive, destination):
    expected = set()
    for split in STORAGE_SPLITS:
        for cls in DRIVE_CLASSES.values():
            prefix = PurePosixPath(split, cls)
            expected.add(prefix / "index.csv")
            for variant in VARIANTS:
                for name in ("signal.npy", "envelope_frames.npy"):
                    expected.add(prefix / variant / name)
    with ZipFile(archive) as z:
        selected = {}
        for info in z.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive member")
            for i, part in enumerate(path.parts):
                if part in STORAGE_SPLITS:
                    rel = PurePosixPath(*path.parts[i:])
                    if rel in expected:
                        if rel in selected:
                            raise ValueError(f"Duplicate archive member for {rel}")
                        selected[rel] = info
                    break
        missing = expected - selected.keys()
        if missing:
            raise ValueError(f"Missing numerical features: {sorted(map(str, missing))}")
        for rel, info in selected.items():
            dst = destination.joinpath(*rel.parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, dst.open("wb") as out:
                shutil.copyfileobj(source, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True, help="Numerical ZIP or extracted folder")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--output", type=Path, default=Path("models/hybrid_experiment"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if a.epochs < 1:
        p.error("--epochs must be positive")
    if not a.check_only and a.output.exists() and any(a.output.iterdir()):
        p.error("Choose a new empty --output directory to avoid mixing experiments")
    with tempfile.TemporaryDirectory(prefix="bearing-hybrid-") as tmp:
        root = a.dataset
        if root.is_file():
            root = Path(tmp)
            extract_needed(a.dataset, root)
        command = [sys.executable, "-m", "src.train_hybrid",
                   "--dataset", str(root), "--raw-dir", str(a.raw_dir),
                   "--epochs", str(a.epochs), "--output-dir", str(a.output)]
        command += ["--check-only"] if a.check_only else ["--baseline"]
        subprocess.run(command, check=True)
        if not a.check_only:
            generate(a.output / "results.json", a.output / "report")
            print(f"Results and discussion: {a.output / 'report' / 'results_and_discussion.md'}")


if __name__ == "__main__":
    main()
