#!/bin/bash
# Deletes what the GitHub runner image preinstalls and the CI jobs never use, so a job fits on
# the runners whose disk is only 72 GB. Prints how long each deletion took and the free space
# before and after.

set -uo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/runner_host.sh"

# Large software of the runner image, relative to the host's root.
unused_host_directories=(
    usr/local/lib/android
    usr/share/dotnet
    usr/local/.ghcup
    opt/hostedtoolcache/CodeQL
)

if ! host_is_mounted; then
    echo "The host's root is not mounted at $host_root; nothing is deleted."
    exit 0
fi

print_group "Free disk space before" df -h "$host_root"
# Keeps the image of the job's own container, which is in use.
SECONDS=0
print_group "Unused Docker images" host_docker image prune --all --force
echo "Removing the unused Docker images took ${SECONDS}s."
for directory in "${unused_host_directories[@]}"; do
    SECONDS=0
    rm -rf "${host_root:?}/$directory"
    echo "Deleting /$directory took ${SECONDS}s."
done
print_group "Free disk space after" df -h "$host_root"
