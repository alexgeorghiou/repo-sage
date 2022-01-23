import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { AddRepoForm } from "./components/AddRepoForm";
import { ChatView } from "./components/ChatView";
import { RepoList } from "./components/RepoList";
import type { ChatMessage, Repo } from "./types";

export default function App() {
  const [repos, setRepos] = useState<Repo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [threads, setThreads] = useState<Record<string, ChatMessage[]>>({});
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await api.listRepos();
      setRepos(list);
      setError(null);
      setSelectedId((cur) => cur ?? list[0]?.id ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const indexing = repos.some((r) => r.state === "indexing" || r.state === "pending");
  useEffect(() => {
    if (!indexing) return;
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [indexing, refresh]);

  const selected = useMemo(() => repos.find((r) => r.id === selectedId) ?? null, [repos, selectedId]);

  async function addRepo(source: string, name?: string, branch?: string) {
    const repo = await api.registerRepo(source, name, branch);
    setSelectedId(repo.id);
    await refresh();
  }

  async function reindex(id: string, full: boolean) {
    try {
      await api.reindex(id, full);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
    await refresh();
  }

  async function remove(id: string) {
    if (!window.confirm(`Remove ${id} and its index?`)) return;
    await api.deleteRepo(id);
    setThreads(({ [id]: _dropped, ...rest }) => rest);
    setSelectedId((cur) => (cur === id ? null : cur));
    await refresh();
  }

  const setMessages = (update: (prev: ChatMessage[]) => ChatMessage[]) => {
    if (!selectedId) return;
    setThreads((t) => ({ ...t, [selectedId]: update(t[selectedId] ?? []) }));
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>
          repo<span>-sage</span>
        </h1>
        <p className="tagline">Chat with any git repository.</p>
        <AddRepoForm onAdd={addRepo} />
        {error && <p className="form-error">{error}</p>}
        <RepoList
          repos={repos}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onReindex={(id, full) => void reindex(id, full)}
          onDelete={(id) => void remove(id)}
        />
      </aside>
      <main className="main">
        {selected ? (
          <ChatView
            key={selected.id}
            repo={selected}
            messages={threads[selected.id] ?? []}
            setMessages={setMessages}
          />
        ) : (
          <div className="welcome">
            <h2>Add a repository to begin</h2>
            <p className="muted">
              Paste a git URL or a path on the API host. Indexing runs in the background and
              later syncs only re-chunk files changed since the last indexed commit.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
