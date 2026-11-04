#!/bin/bash
# Prints the machine a CI job runs on and how much disk, memory and CPU it has left, so a
# job that runs out of space or behaves differently on one runner can be traced back to it.

set -uo pipefail

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
print_group "Disk space" df -h
print_group "Inodes" df -ih
print_group "Directory sizes" bash -c 'du -xsh "$@" 2>/dev/null | sort -h' _ "${measured_directories[@]}"
