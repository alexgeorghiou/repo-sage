import type { JobReceipt, JobRequest } from "./types";

export class QueueClientError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
  }
}

/** Submits jobs to a tinyqueue HTTP gateway. */
export class QueueClient {
  constructor(
    private readonly baseUrl: string,
    private readonly timeoutMs = 5000,
  ) {}

  async submit(job: JobRequest): Promise<JobReceipt> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(`${this.baseUrl}/jobs`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(job),
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new QueueClientError(res.status, await res.text());
      }
      return (await res.json()) as JobReceipt;
    } finally {
      clearTimeout(timer);
    }
  }
}

export const withPriority = (job: JobRequest, priority: number): JobRequest => ({
  ...job,
  priority,
});
