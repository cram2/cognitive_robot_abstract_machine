# Notification Retry Overhaul — Roadmap

This is a **fictional example plan**, written to accompany
`example-walkthrough.md`. Nothing here describes a real initiative, branch,
or pull request.

## Why

Notification delivery currently fails hard on the first transient error.
This plan introduces a retry/backoff mechanism, a circuit breaker so a
persistently failing downstream service doesn't get hammered, and the
metrics/alerting needed to operate it safely.

## Decisions locked in

- Retries use exponential backoff with jitter, not a fixed delay.
- Every item in the `retry-logic` track stacks its branch directly on
  `retry-backoff-strategy`'s branch once that pull request is open and
  ready for review — no need to wait for it to merge first.
- A message that exhausts its retries goes to a dead-letter queue rather
  than being dropped silently.

## Next steps

Once `retry-circuit-breaker` is out of draft, the observability track can
start drafting alert thresholds against its real event shape.
