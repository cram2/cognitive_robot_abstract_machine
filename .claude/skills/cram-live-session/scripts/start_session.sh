#!/usr/bin/env bash
# Start the live-session kernel inside this container, unless it runs already. Run it
# detached, e.g.
#   docker exec -d CONTAINER bash /opt/cram/.claude/skills/cram-live-session/scripts/start_session.sh
# Settings come from the environment; the defaults fit the CRAM docker image.
set -euo pipefail

REPOSITORY=${LIVE_REPOSITORY:-/opt/cram}
ENVIRONMENT_SCRIPT=${LIVE_ENVIRONMENT_SCRIPT:-/entrypoint.sh}
SESSION_DIRECTORY=${LIVE_SESSION_DIRECTORY:-/root/work/live_session}
MEMORY_LIMIT_KB=${LIVE_MEMORY_LIMIT_KB:-150000000}

mkdir -p "$SESSION_DIRECTORY"
cd "$REPOSITORY"
# ROS setup scripts read variables that may be unset
set +u
source "$ENVIRONMENT_SCRIPT"
set -u
export QT_QPA_PLATFORM=offscreen

if ! pgrep -f "^[^ ]*python[^ ]* -m ipykernel_launcher -f $SESSION_DIRECTORY/kernel.json" > /dev/null; then
    (
        ulimit -v "$MEMORY_LIMIT_KB"
        exec python -m ipykernel_launcher -f "$SESSION_DIRECTORY/kernel.json" \
            > "$SESSION_DIRECTORY/kernel.log" 2>&1
    ) &
fi
wait
