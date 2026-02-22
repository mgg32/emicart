import ctypes
import os
from pathlib import Path
import re
from typing import Optional


def get_default_save_dir() -> Path:
    home = Path.home()
    documents_dir = home / "Documents"
    save_path = documents_dir / "EmiCart" / "outputs"
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def get_usb_save_dir() -> Optional[Path]:
    if os.name == "nt":
        return _get_usb_save_dir_windows()
    return _get_usb_save_dir_linux()


def _get_usb_save_dir_windows() -> Optional[Path]:
    try:
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        get_drive_type = ctypes.windll.kernel32.GetDriveTypeW
    except Exception:
        return None

    removable_roots = []
    for i in range(26):
        if not (bitmask & (1 << i)):
            continue
        letter = chr(ord("A") + i)
        root = Path(f"{letter}:\\")
        try:
            # DRIVE_REMOVABLE = 2
            if int(get_drive_type(str(root))) == 2:
                removable_roots.append(root)
        except Exception:
            continue

    for root in removable_roots:
        try:
            save_path = root / "EmiCart" / "outputs"
            save_path.mkdir(parents=True, exist_ok=True)
            return save_path
        except Exception:
            continue
    return None


def _get_usb_save_dir_linux() -> Optional[Path]:
    user = os.environ.get("USER", "").strip()
    search_roots = []
    if user:
        search_roots.append(Path("/media") / user)
        search_roots.append(Path("/run/media") / user)
    search_roots.append(Path("/media"))
    search_roots.append(Path("/run/media"))
    search_roots.append(Path("/mnt"))

    for base in search_roots:
        if not base.exists() or not base.is_dir():
            continue
        try:
            candidates = sorted([p for p in base.iterdir() if p.is_dir()])
        except Exception:
            continue
        for mount in candidates:
            try:
                save_path = mount / "EmiCart" / "outputs"
                save_path.mkdir(parents=True, exist_ok=True)
                return save_path
            except Exception:
                continue
    return None


def normalize_stem(name: str) -> str:
    stem = (name or "").strip()
    if not stem:
        return "output"
    return Path(stem).stem


def split_base_and_index(stem: str):
    m = re.match(r"^(.*)_(\d+)$", stem)
    if not m:
        return stem, None
    return m.group(1), int(m.group(2))


def next_available_stem(save_dir: Path, base_stem: str) -> str:
    requested = normalize_stem(base_stem)
    base, _ = split_base_and_index(requested)

    def is_stem_taken(stem: str) -> bool:
        for p in save_dir.iterdir():
            if not p.is_file():
                continue
            if p.stem == stem:
                return True
        return False

    if requested == base and not is_stem_taken(base):
        return base

    used = set()
    if is_stem_taken(base):
        used.add(0)

    for p in save_dir.glob(f"{base}_*"):
        if not p.is_file():
            continue
        stem = p.stem
        b, idx = split_base_and_index(stem)
        if b == base and idx is not None:
            used.add(idx)

    idx = 1
    while idx in used:
        idx += 1
    return f"{base}_{idx:03d}"
