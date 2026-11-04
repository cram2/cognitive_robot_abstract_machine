#!/bin/bash
# The host a CI job's container runs on, for the scripts that inspect it or free its disk.
# Sourced, not run.

# Where the CI workflows mount the host's root into the job's container.
host_root=/host

# Prints the output of a command as a collapsible group of the job log.
print_group() {
    local title="$1"
    shift
    echo "::group::$title"
    "$@" 2>&1
    echo "::endgroup::"
}

# Whether the host's root is mounted into this container.
host_is_mounted() {
    [[ -d "$host_root/usr" ]]
}

# Runs the host's Docker client against the socket the runner mounts into the container.
# The client runs with the host's own C library, which can be newer than the one in the image.
host_docker() {
    "$host_root/lib64/ld-linux-x86-64.so.2" --library-path "$host_root/lib/x86_64-linux-gnu" \
        "$host_root/usr/bin/docker" "$@"
}
