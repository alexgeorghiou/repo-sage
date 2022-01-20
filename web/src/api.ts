import type { AnswerPayload, Repo, Snippet, Source } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  listRepos: () => request<Repo[]>("/api/repos"),
  getRepo: (id: string) => request<Repo>(`/api/repos/${encodeURIComponent(id)}`),
  registerRepo: (source: string, name?: string, branch?: string) =>
    request<Repo>("/api/repos", {
      method: "POST",
      body: JSON.stringify({ source, name: name || null, branch: branch || null }),
    }),
  reindex: (id: string, full = false) =>
    request<Repo>(`/api/repos/${encodeURIComponent(id)}/reindex`, {
      method: "POST",
      body: JSON.stringify({ full }),
    }),
  deleteRepo: (id: string) =>
    request<void>(`/api/repos/${encodeURIComponent(id)}`, { method: "DELETE" }),
  snippet: (id: string, path: string, start: number, end: number, context = 3) => {
    const q = new URLSearchParams({
      path,
      start: String(start),
      end: String(end),
      context: String(context),
    });
    return request<Snippet>(`/api/repos/${encodeURIComponent(id)}/snippet?${q.toString()}`);
  },
};

export interface StreamHandlers {
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onAnswer: (answer: AnswerPayload) => void;
  onError: (message: string) => void;
}

interface SseEvent {
  event: string;
  data: string;
}

function parseEvents(buffer: string): { events: SseEvent[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: SseEvent[] = [];
  for (const block of blocks) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (data.length > 0) events.push({ event, data: data.join("\n") });
  }
  return { events, rest };
}

/** POST a question and consume the server-sent event stream it returns. */
export async function askStream(
  repoId: string,
  question: string,
  history: { role: "user" | "assistant"; content: string }[],
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/api/repos/${encodeURIComponent(repoId)}/ask/stream`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({ question, history }),
    signal,
  });
  if (!res.ok || !res.body) {
    handlers.onError(`${res.status}: ${res.statusText}`);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseEvents(buffer);
    buffer = parsed.rest;
    for (const ev of parsed.events) {
      const payload: unknown = JSON.parse(ev.data);
      switch (ev.event) {
        case "sources":
          handlers.onSources(payload as Source[]);
          break;
        case "token":
          handlers.onToken(payload as string);
          break;
        case "answer":
          handlers.onAnswer(payload as AnswerPayload);
          break;
        case "error":
          handlers.onError(String(payload));
          break;
        default:
          break;
      }
    }
  }
}
