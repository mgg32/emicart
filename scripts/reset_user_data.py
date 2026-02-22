from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _paths() -> dict[str, Path]:
    base = Path.home() / "Documents" / "EmiCart"
    return {
        "base": base,
        "limits": base / "limits.json",
        "probes": base / "probes.json",
        "outputs": base / "outputs",
    }


def _remove_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset EmiCart user data under ~/Documents/EmiCart."
    )
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="Also delete the outputs directory.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    p = _paths()
    to_delete = [p["limits"], p["probes"]]
    if args.include_outputs:
        to_delete.append(p["outputs"])

    if not args.yes:
        print("About to delete:")
        for item in to_delete:
            print(f"  - {item}")
        answer = input("Continue? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    deleted_any = False
    for item in to_delete:
        deleted = _remove_path(item)
        print(f"{'Deleted' if deleted else 'Not found'}: {item}")
        deleted_any = deleted_any or deleted

    base = p["base"]
    try:
        if base.exists() and not any(base.iterdir()):
            base.rmdir()
            print(f"Deleted empty directory: {base}")
    except Exception:
        pass

    if not deleted_any:
        print("No matching EmiCart user data was found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
