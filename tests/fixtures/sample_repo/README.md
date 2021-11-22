# tinyqueue

A deliberately small background job queue used as the evaluation fixture for repo-sage.

## Concepts

- **JobQueue** keeps pending jobs in a priority heap; lower numbers run first.
- **Worker** pulls jobs, runs the registered handler and acknowledges or retries them.
- **RetryPolicy** decides how long to wait before a failed job is attempted again.
- **Scheduler** enqueues periodic jobs on a fixed interval.
- **JsonlStorage** persists jobs to an append-only JSON-lines file.

## Running a worker

```bash
python -m tinyqueue.worker --queue default --concurrency 4
```

Send `SIGTERM` to drain in-flight jobs and exit cleanly.

## TypeScript client

`client/src/client.ts` submits jobs over HTTP with a request timeout.
