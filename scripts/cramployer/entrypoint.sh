#!/usr/bin/env bash
# %% Application environment
set -eo pipefail
source /deployment/environment.sh
exec "$@"
