export interface JobRequest {
  name: string;
  payload?: Record<string, unknown>;
  priority?: number;
  delaySeconds?: number;
}

export interface JobReceipt {
  id: string;
  enqueuedAt: string;
}
