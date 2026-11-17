#!/bin/bash
# Deletes what the GitHub runner image preinstalls and the CI jobs never use, so a job fits on
# the runners whose disk is only 72 GB. The quick deletions are done when this script returns;
# the slow ones go on in the background while the job installs its dependencies and write how
# long they took to the log show_runner_resources.sh prints at the end of the job.

set -uo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/runner_host.sh"

# Software of the runner image, relative to the host's root, that takes seconds to delete.
quickly_deleted_directories=(
    usr/share/dotnet
    usr/local/.ghcup
    opt/hostedtoolcache/CodeQL
)

# Software of the runner image that holds so many files that deleting it takes about a minute.
slowly_deleted_directories=(
    usr/local/lib/android
)

# How many rm processes delete one directory at once. The runner's disk serves concurrent
# requests much faster than one rm deleting file after file.
parallel_deletions=16

# Deletes a directory of the host, with the subtrees three levels down deleted in parallel.
delete_host_directory() {
    local directory="$host_root/$1"
    local started=$SECONDS
    find "$directory" -mindepth 3 -maxdepth 3 -print0 2>/dev/null |
        xargs -0 -r -P "$parallel_deletions" -n 1 rm -rf
    rm -rf "$directory"
    echo "Deleting /$1 took $((SECONDS - started))s."
}

# Removes every Docker image no container uses, which keeps the image of the job's container.
remove_unused_docker_images() {
    local started=$SECONDS
    host_docker image prune --all --force | tail -n 1
    echo "Removing the unused Docker images took $((SECONDS - started))s."
}

if ! host_is_mounted; then
    echo "The host's root is not mounted at $host_root; nothing is deleted."
    exit 0
fi

print_group "Free disk space before" df -h "$host_root"
remove_unused_docker_images &
for directory in "${quickly_deleted_directories[@]}"; do
    delete_host_directory "$directory" &
done
wait
print_group "Free disk space after the quick deletions" df -h "$host_root"

# Detached from this step's output, so the step ends without waiting for the deletion.
(
    for directory in "${slowly_deleted_directories[@]}"; do
        delete_host_directory "$directory"
    done
    df -h "$host_root"
) >"$background_deletion_log" 2>&1 </dev/null &
disown
echo "Deleting ${slowly_deleted_directories[*]} in the background, logging to $background_deletion_log."
