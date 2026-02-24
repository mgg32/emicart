#!/usr/bin/env bash
set -euo pipefail

# Install EmiCart on Raspberry Pi (Debian/Raspberry Pi OS).
# Usage:
#   ./scripts/install_pi.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v apt >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y \
    build-essential \
    pkg-config \
    python3-dev \
    python3-venv \
    python3-pip \
    python3-tk \
    libopenblas0 \
    libatlas3-base \
    libatlas-base-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libwebp-dev \
    libharfbuzz-dev \
    libfribidi-dev
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

# Prefer Raspberry Pi wheels when available.
PIWHEELS_URL="https://www.piwheels.org/simple"

# Install Pillow first (binary wheel only) since matplotlib depends on it.
# Try a few versions that are commonly available on Bullseye-era Pi builds.
PILLOW_OK=0
for PILLOW_VER in 10.4.0 10.3.0 10.2.0 9.5.0; do
  if python -m pip install --only-binary=:all: --extra-index-url "$PIWHEELS_URL" "Pillow==${PILLOW_VER}"; then
    PILLOW_OK=1
    break
  fi
done
if [ "$PILLOW_OK" -ne 1 ]; then
  echo "ERROR: Could not install a binary Pillow wheel from piwheels."
  echo "Try updating Raspberry Pi OS / Python, or use wheel-bundle deployment."
  exit 1
fi

# Runtime deps used by the GUI/instrument path.
python -m pip install --prefer-binary --extra-index-url "$PIWHEELS_URL" numpy "matplotlib<3.10" scipy pyvisa pyvisa-py pyusb psutil zeroconf

# Install project package.
python -m pip install .

# Create desktop launcher(s).
echo "Creating desktop launcher..."
DESKTOP_FILE_CONTENT="[Desktop Entry]
Type=Application
Name=EmiCart
Comment=Launch EmiCart GUI
Exec=$REPO_ROOT/.venv/bin/python -m emicart
Path=$HOME
Terminal=false
Icon=$REPO_ROOT/emicart/data/nasa_meatball.png
StartupNotify=true
Categories=Science;Utility;
"

launcher_created=0

for desktop_dir in "$HOME/Desktop" "$HOME/.local/share/applications"; do
  mkdir -p "$desktop_dir"
  desktop_file="$desktop_dir/EmiCart.desktop"
  printf "%s" "$DESKTOP_FILE_CONTENT" > "$desktop_file"
  chmod +x "$desktop_file"
  if command -v gio >/dev/null 2>&1; then
    gio set "$desktop_file" "metadata::trusted" true >/dev/null 2>&1 || true
  fi
  echo "Created launcher: $desktop_file"
  launcher_created=1
done

if [ "$launcher_created" -eq 0 ]; then
  echo "Could not create a launcher automatically."
fi

echo
echo "Install complete."
echo "Activate env: source .venv/bin/activate"
echo "Run GUI:      python -m emicart"
