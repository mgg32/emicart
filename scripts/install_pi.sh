#!/usr/bin/env bash
set -euo pipefail

# Install EmiCart on Raspberry Pi (Debian/Raspberry Pi OS).
# Usage:
#   ./scripts/install_pi.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

USBTMC_RULE_PATH="/etc/udev/rules.d/99-usbtmc.rules"
USBTMC_RULE='SUBSYSTEM=="usb", ATTRS{idVendor}=="0699", MODE="0666", GROUP="plugdev", RUN+="/bin/sh -c '\''echo %k > /sys/bus/usb/drivers/usbtmc/unbind'\''"'

run_privileged() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi

  echo "ERROR: This step requires elevated privileges and sudo is unavailable."
  echo "Manual fallback command:"
  echo "  echo \"$USBTMC_RULE\" | tee $USBTMC_RULE_PATH >/dev/null"
  return 1
}

install_usbtmc_rule() {
  echo "Installing USBTMC udev rule..."

  local existing=""
  if run_privileged test -f "$USBTMC_RULE_PATH"; then
    existing="$(run_privileged cat "$USBTMC_RULE_PATH")"
  fi

  if [ "$existing" != "$USBTMC_RULE" ]; then
    printf "%s\n" "$USBTMC_RULE" | run_privileged tee "$USBTMC_RULE_PATH" >/dev/null
    echo "Installed udev rule at $USBTMC_RULE_PATH"
  else
    echo "Udev rule already up to date at $USBTMC_RULE_PATH"
  fi

  run_privileged udevadm control --reload-rules
  run_privileged udevadm trigger
  echo "Reloaded udev rules and triggered devices."

  if command -v modprobe >/dev/null 2>&1 && lsmod | awk '{print $1}' | grep -qx usbtmc; then
    run_privileged modprobe -r usbtmc
    run_privileged modprobe usbtmc
    echo "Reloaded usbtmc kernel module."
  fi
}

install_usbtmc_rule

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
