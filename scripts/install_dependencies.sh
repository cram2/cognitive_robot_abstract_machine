#!/usr/bin/env bash
# %% Installation options
# Usage: bash scripts/install_dependencies.sh {cpu|default} [robokudo,semantic_digital_twin]
set -euo pipefail
torch_mode="${1:-cpu}"
optional_packages="${2:-}"
case "$torch_mode" in
    cpu|default) ;;
    *) printf 'Unsupported PyTorch mode: %s\n' "$torch_mode" >&2; exit 2 ;;
esac
IFS=, read -r -a selected_packages <<< "$optional_packages"
for package in "${selected_packages[@]}"; do
    case "$package" in
        robokudo|semantic_digital_twin|'') ;;
        *) printf 'Unsupported optional package: %s\n' "$package" >&2; exit 2 ;;
    esac
done
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/opt/ros/.uv_cache}"

# %% Common dependencies
privilege_command=()
if (( EUID != 0 )); then
    privilege_command=(sudo)
fi
"${privilege_command[@]}" apt-get update
"${privilege_command[@]}" apt-get install -y xvfb libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 \
    libxkbcommon-x11-0 libegl1
uv sync --extra dev --active
uv tree --depth 1

# %% Optional dependencies
torch_index=()
if [[ "$torch_mode" == cpu ]]; then
    torch_index=(--extra-index-url https://download.pytorch.org/whl/cpu)
fi
for package in "${selected_packages[@]}"; do
    case "$package" in
        robokudo)
            # Preserve editable workspace sources while installing vision dependencies.
            uv pip install -r robokudo/pyproject.toml --extra dev --no-sources "${torch_index[@]}"
            ;;
        semantic_digital_twin) uv pip install drake ;;
    esac
done
