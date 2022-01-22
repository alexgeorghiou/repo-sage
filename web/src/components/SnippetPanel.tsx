import { useEffect, useState } from "react";
import { api } from "../api";
import { citationLabel, type ParsedCitation } from "../citations";
import type { Snippet } from "../types";

interface Props {
  repoId: string;
  citation: ParsedCitation;
  onClose: () => void;
}

export function SnippetPanel({ repoId, citation, onClose }: Props) {
  const [snippet, setSnippet] = useState<Snippet | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setSnippet(null);
    setError(null);
    api
      .snippet(repoId, citation.path, citation.start, citation.end, 4)
      .then((s) => !cancelled && setSnippet(s))
      .catch((e: unknown) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [repoId, citation.path, citation.start, citation.end]);

  return (
    <aside className="snippet-panel">
      <header>
        <code>{citationLabel(citation)}</code>
        <button onClick={onClose} aria-label="Close">
          ×
        </button>
      </header>
      {error && <p className="form-error">{error}</p>}
      {!snippet && !error && <p className="muted">Loading…</p>}
      {snippet && (
        <pre className="code">
          {snippet.content.split("\n").map((line, i) => {
            const n = snippet.start_line + i;
            const hit = n >= citation.start && n <= citation.end;
            return (
              <div key={n} className={hit ? "line hit" : "line"}>
                <span className="ln">{n}</span>
                <span>{line || " "}</span>
              </div>
            );
          })}
        </pre>
      )}
    </aside>
  );
}
