import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { askStream } from "../api";
import { citationKey, splitAnswer, type ParsedCitation } from "../citations";
import type { ChatMessage, Repo } from "../types";
import { CitationChip } from "./CitationChip";
import { SnippetPanel } from "./SnippetPanel";

interface Props {
  repo: Repo;
  messages: ChatMessage[];
  setMessages: (update: (prev: ChatMessage[]) => ChatMessage[]) => void;
}

const SUGGESTIONS = [
  "Give me a tour of the main modules.",
  "Where is error handling and retry logic?",
  "How is configuration loaded?",
];

let counter = 0;
const nextId = () => `m${Date.now()}-${counter++}`;

function AnswerBody({
  message,
  activeKey,
  onOpen,
}: {
  message: ChatMessage;
  activeKey: string | null;
  onOpen: (c: ParsedCitation) => void;
}) {
  const invalid = new Set(
    (message.answer?.invalid_citations ?? []).map((c) => `${c.path}:${c.start_line}-${c.end_line}`),
  );
  return (
    <div className="answer">
      {splitAnswer(message.content).map((part, i) =>
        part.kind === "text" ? (
          <span key={i}>{part.text}</span>
        ) : (
          <span key={i} className="chips">
            {part.items.map((c) => (
              <CitationChip
                key={citationKey(c)}
                citation={c}
                verified={!message.answer || !invalid.has(citationKey(c))}
                active={activeKey === citationKey(c)}
                onOpen={onOpen}
              />
            ))}
          </span>
        ),
      )}
      {message.pending && <span className="cursor" />}
    </div>
  );
}

export function ChatView({ repo, messages, setMessages }: Props) {
  const [input, setInput] = useState("");
  const [open, setOpen] = useState<ParsedCitation | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const busy = messages.some((m) => m.pending);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => () => abortRef.current?.abort(), [repo.id]);

  async function send(question: string) {
    if (!question.trim() || busy) return;
    const history = messages
      .filter((m) => !m.pending && !m.error)
      .map((m) => ({ role: m.role, content: m.content }));
    const assistantId = nextId();
    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: "user", content: question },
      { id: assistantId, role: "assistant", content: "", pending: true },
    ]);
    setInput("");
    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await askStream(
        repo.id,
        question,
        history,
        {
          onSources: (sources) => patch((m) => ({ ...m, sources })),
          onToken: (text) => patch((m) => ({ ...m, content: m.content + text })),
          onAnswer: (answer) =>
            patch((m) => ({ ...m, content: answer.answer, answer, pending: false })),
          onError: (error) => patch((m) => ({ ...m, error, pending: false })),
        },
        controller.signal,
      );
    } catch (err) {
      if (!controller.signal.aborted) {
        patch((m) => ({ ...m, error: err instanceof Error ? err.message : String(err) }));
      }
    } finally {
      patch((m) => ({ ...m, pending: false }));
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void send(input);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send(input);
    }
  }

  const activeKey = open ? citationKey(open) : null;

  return (
    <div className={open ? "chat with-panel" : "chat"}>
      <section className="thread">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>Ask anything about {repo.name}</h2>
            <p className="muted">
              Answers are grounded in retrieved code and cite <code>file:lines</code>. Click a
              citation to see the exact lines.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => void send(s)} disabled={repo.state !== "ready"}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m) => (
          <article key={m.id} className={`msg msg-${m.role}`}>
            {m.role === "user" ? (
              <p>{m.content}</p>
            ) : (
              <>
                <AnswerBody message={m} activeKey={activeKey} onOpen={setOpen} />
                {m.error && <p className="form-error">{m.error}</p>}
                {m.answer && !m.answer.grounded && m.answer.citations.length === 0 && (
                  <p className="muted small">No verifiable citations in this answer.</p>
                )}
                {m.sources && m.sources.length > 0 && (
                  <details className="sources">
                    <summary>{m.sources.length} retrieved chunks</summary>
                    <ol>
                      {m.sources.map((s) => (
                        <li key={`${s.path}:${s.start_line}`}>
                          <button
                            className="link"
                            onClick={() =>
                              setOpen({ path: s.path, start: s.start_line, end: s.end_line })
                            }
                          >
                            {s.path}:{s.start_line}-{s.end_line}
                          </button>
                          {s.symbol && <code className="symbol">{s.symbol}</code>}
                          <span className="muted small">
                            {Object.entries(s.retrievers)
                              .map(([name, rank]) => `${name} #${rank}`)
                              .join(" · ")}
                          </span>
                        </li>
                      ))}
                    </ol>
                  </details>
                )}
              </>
            )}
          </article>
        ))}
        <div ref={bottomRef} />
      </section>
      {open && <SnippetPanel repoId={repo.id} citation={open} onClose={() => setOpen(null)} />}
      <form className="composer" onSubmit={onSubmit}>
        <textarea
          rows={2}
          value={input}
          placeholder={
            repo.state === "ready" ? `Ask about ${repo.name}…` : `Waiting for ${repo.name} to index…`
          }
          disabled={repo.state !== "ready"}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button type="submit" disabled={busy || !input.trim() || repo.state !== "ready"}>
          {busy ? "Thinking…" : "Ask"}
        </button>
      </form>
    </div>
  );
}
