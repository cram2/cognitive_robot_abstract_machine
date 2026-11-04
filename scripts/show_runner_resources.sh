#!/bin/bash
# Prints the machine a CI job runs on and how much disk, memory and CPU it has left, so a
# job that runs out of space or behaves differently on one runner can be traced back to it.
# Usage: show_runner_resources.sh [--host-directory-sizes]
#   --host-directory-sizes  also measures the largest directories on the host's disk, which
#                           takes a minute or two.

set -uo pipefail

# Where the CI workflows mount the host's root into the job's container.
host_root=/host

# Where the CI image keeps its environments and caches, and where the job writes.
measured_directories=(/opt/ros/cram-env /opt/ros/.uv_cache /opt/ros/overlay_ws /github/home /__w /tmp /usr /var)

# Prints the output of a command as a collapsible group of the job log.
print_group() {
    local title="$1"
    shift
    echo "::group::$title"
    "$@" 2>&1
    echo "::endgroup::"
}

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

if [[ ! -d "$host_root/usr" ]]; then
    echo "The host's root is not mounted at $host_root; skipping the host's disk usage."
    exit 0
fi
# The images, containers and build cache Docker keeps on the host, read through the socket
# the runner mounts into the container. The host's Docker client runs with the host's own C
# library, which can be newer than the one in the image.
print_group "Docker disk usage" "$host_root/lib64/ld-linux-x86-64.so.2" \
    --library-path "$host_root/lib/x86_64-linux-gnu" "$host_root/usr/bin/docker" system df -v
if [[ "${1:-}" == "--host-directory-sizes" ]]; then
    print_group "Host directory sizes" bash -c \
        'du -xh --max-depth=3 --threshold=500M "$1" 2>/dev/null | sort -h | tail -n 60' _ "$host_root"
fi
