#!/bin/bash
# Prints the machine a CI job runs on and how much disk, memory and CPU it has left, so a
# job that runs out of space or behaves differently on one runner can be traced back to it.
# Usage: show_runner_resources.sh [--host-directory-sizes]
#   --host-directory-sizes  also measures the largest directories on the host's disk, which
#                           takes a minute or two.

set -uo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/runner_host.sh"

# Where the CI image keeps its environments and caches, and where the job writes.
measured_directories=(/opt/ros/cram-env /opt/ros/.uv_cache /opt/ros/overlay_ws /github/home /__w /tmp /usr /var)

if [[ -f "$background_deletion_log" ]]; then
    print_group "Deletions in the background" cat "$background_deletion_log"
fi
print_group "Operating system" bash -c 'uname -a; cat /etc/os-release'
# The runners differ in CPU model, which can change the rounding of numeric results.
print_group "CPU" lscpu
print_group "Memory" free -h
print_group "Swap" cat /proc/swaps
# Leaves out the loop devices snaps are mounted from.
print_group "Block devices" lsblk --exclude 7 -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
print_group "Disk space" df -h
print_group "Inodes" df -ih
print_group "Directory sizes" bash -c 'du -xsh "$@" 2>/dev/null | sort -h' _ "${measured_directories[@]}"

if ! host_is_mounted; then
    echo "The host's root is not mounted at $host_root; skipping the host's disk usage."
    exit 0
fi
# The images, containers and build cache Docker keeps on the host.
print_group "Docker disk usage" host_docker system df -v
if [[ "${1:-}" == "--host-directory-sizes" ]]; then
    print_group "Host directory sizes" bash -c \
        'du -xh --max-depth=3 --threshold=500M "$1" 2>/dev/null | sort -h | tail -n 60' _ "$host_root"
fi
