export type IndexState = "pending" | "indexing" | "ready" | "failed";

export interface IndexReport {
  mode: "full" | "incremental" | "up-to-date";
  commit: string | null;
  previous_commit: string | null;
  files_indexed: number;
  files_deleted: number;
  chunks_written: number;
  skipped: number;
}

export interface Repo {
  id: string;
  name: string;
  source: string;
  branch: string | null;
  last_commit: string | null;
  indexed_at: string | null;
  file_count: number;
  chunk_count: number;
  state: IndexState;
  error: string | null;
  last_report: IndexReport | null;
}

export interface Citation {
  path: string;
  start_line: number;
  end_line: number;
  label: string;
}

export interface Source {
  path: string;
  start_line: number;
  end_line: number;
  symbol: string | null;
  kind: string;
  language: string | null;
  score: number;
  retrievers: Record<string, number>;
  preview: string;
}

export interface AnswerPayload {
  question: string;
  answer: string;
  citations: Citation[];
  invalid_citations: Citation[];
  grounded: boolean;
}

export interface Snippet {
  path: string;
  start_line: number;
  end_line: number;
  language: string | null;
  content: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  answer?: AnswerPayload;
  pending?: boolean;
  error?: string;
}
