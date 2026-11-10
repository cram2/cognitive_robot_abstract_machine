#!/usr/bin/env bash
# Drive the live-session kernel in a container on a remote server from this machine.
#
#   live_kernel.sh perform FILE [FIRST:LAST]  perform FILE's cells through the session
#   live_kernel.sh raw FILE                   run FILE as it is, outside the session
#   live_kernel.sh wait NAME                  keep following the log of an earlier run
#   live_kernel.sh status                     whether the kernel computes or sleeps
#   live_kernel.sh interrupt                  KeyboardInterrupt in the kernel
#
# Commands run detached inside the container and write a log to the exchange directory,
# because an attached `docker exec` can hang on exit. Settings come from the environment:
# LIVE_SERVER, LIVE_CONTAINER and LIVE_EXCHANGE_ON_SERVER (the server's side of a directory
# mounted into the container at LIVE_EXCHANGE_IN_CONTAINER) have no default.
set -euo pipefail

SERVER=${LIVE_SERVER:?set LIVE_SERVER to the ssh destination of the server}
CONTAINER=${LIVE_CONTAINER:?set LIVE_CONTAINER to the container the kernel runs in}
EXCHANGE_ON_SERVER=${LIVE_EXCHANGE_ON_SERVER:?set LIVE_EXCHANGE_ON_SERVER to the server path mounted at LIVE_EXCHANGE_IN_CONTAINER}
EXCHANGE_IN_CONTAINER=${LIVE_EXCHANGE_IN_CONTAINER:-/root/live_logs}
REPOSITORY_IN_CONTAINER=${LIVE_REPOSITORY_IN_CONTAINER:-/opt/cram}
ENVIRONMENT_SCRIPT=${LIVE_ENVIRONMENT_SCRIPT:-/entrypoint.sh}
CONNECTION_FILE=${LIVE_CONNECTION_FILE:-/root/work/live_session/kernel.json}
WAIT_SECONDS=${LIVE_WAIT_SECONDS:-540}
SILENCE_SECONDS=${LIVE_SILENCE_SECONDS:-90}
PERFORM_CELLS=$REPOSITORY_IN_CONTAINER/.claude/skills/cram-live-session/scripts/perform_cells.py

kernel_pid_on_server() {
    ssh "$SERVER" "docker top $CONTAINER -o pid,args" \
        | awk -v file="$CONNECTION_FILE" 'index($0, "-m ipykernel_launcher -f " file) && !/bash -c/ {print $1; exit}'
}

cpu_ticks_in_five_seconds() {
    local pid=$1
    ssh "$SERVER" "a=\$(awk '{print \$14+\$15}' /proc/$pid/stat); sleep 5; \
        b=\$(awk '{print \$14+\$15}' /proc/$pid/stat); echo \$((b-a))"
}

# Print the log of a run as it grows, until it ends, WAIT_SECONDS pass, or the kernel
# sleeps through SILENCE_SECONDS without output (a deadlock, most likely).
follow() {
    local name=$1 log="$EXCHANGE_ON_SERVER/$1.log" printed=0 silent=0 waited=0
    while (( waited < WAIT_SECONDS )); do
        local lines
        lines=$(ssh "$SERVER" "tail -n +$((printed + 1)) $log 2>/dev/null || true" | sed 's/\x1b\[[0-9;]*m//g')
        if [[ -n $lines ]]; then
            printf '%s\n' "$lines"
            printed=$((printed + $(printf '%s\n' "$lines" | wc -l)))
            silent=0
            grep -q '^EXIT ' <<<"$lines" && return 0
        else
            silent=$((silent + 5))
        fi
        if (( silent >= SILENCE_SECONDS )); then
            local pid ticks
            pid=$(kernel_pid_on_server)
            ticks=$(cpu_ticks_in_five_seconds "$pid")
            if (( ticks < 20 )); then
                echo "[kernel sleeps without output for ${silent} s ($ticks CPU ticks in 5 s): probably deadlocked; 'live_kernel.sh interrupt' shows where]"
                return 2
            fi
            silent=0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "[still running after ${WAIT_SECONDS} s; continue with 'live_kernel.sh wait $name']"
    return 3
}

run_detached() {
    local name=$1; shift
    ssh "$SERVER" "rm -f $EXCHANGE_ON_SERVER/$name.log; docker exec -d $CONTAINER bash -c \
        'cd $REPOSITORY_IN_CONTAINER && source $ENVIRONMENT_SCRIPT && \
         python -u $PERFORM_CELLS --connection-file $CONNECTION_FILE $* > $EXCHANGE_IN_CONTAINER/$name.log 2>&1; \
         echo EXIT \$? >> $EXCHANGE_IN_CONTAINER/$name.log'"
    follow "$name"
}

command=${1:-}
case $command in
    perform | raw)
        file=$2
        name=$(basename "$file" .py)
        scp -q "$file" "$SERVER:$EXCHANGE_ON_SERVER/$name.py"
        if [[ $command == raw ]]; then
            run_detached "$name" "$EXCHANGE_IN_CONTAINER/$name.py" --raw
        else
            run_detached "$name" "$EXCHANGE_IN_CONTAINER/$name.py" --cells "${3:-:}"
        fi
        ;;
    wait)
        follow "$2"
        ;;
    status)
        pid=$(kernel_pid_on_server)
        echo "kernel pid $pid on the server: $(cpu_ticks_in_five_seconds "$pid") CPU ticks in 5 s"
        ;;
    interrupt)
        ssh "$SERVER" "docker exec -d $CONTAINER bash -c \
            'kill -INT \$(pgrep -f \"^[^ ]*python[^ ]* -m ipykernel_launcher -f $CONNECTION_FILE\")'"
        echo "interrupt sent"
        ;;
    *)
        sed -n '2,14p' "$0"
        exit 1
        ;;
esac
