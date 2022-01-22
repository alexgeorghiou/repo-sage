import type { Repo } from "../types";

interface Props {
  repos: Repo[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onReindex: (id: string, full: boolean) => void;
  onDelete: (id: string) => void;
}

function relativeTime(iso: string | null): string {
  if (!iso) return "never";
  const seconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function RepoList({ repos, selectedId, onSelect, onReindex, onDelete }: Props) {
  if (repos.length === 0) {
    return <p className="empty">No repositories yet. Add one above to start.</p>;
  }
  return (
    <ul className="repo-list">
      {repos.map((repo) => (
        <li
          key={repo.id}
          className={repo.id === selectedId ? "repo selected" : "repo"}
          onClick={() => onSelect(repo.id)}
        >
          <div className="repo-head">
            <span className="repo-name" title={repo.source}>
              {repo.name}
            </span>
            <span className={`badge badge-${repo.state}`}>{repo.state}</span>
          </div>
          <div className="repo-meta">
            <code>{repo.last_commit ? repo.last_commit.slice(0, 7) : "working tree"}</code>
            <span>
              {repo.file_count} files · {repo.chunk_count} chunks
            </span>
            <span>indexed {relativeTime(repo.indexed_at)}</span>
          </div>
          {repo.last_report && (
            <div className="repo-report">
              last run: {repo.last_report.mode}, {repo.last_report.files_indexed} files re-chunked
              {repo.last_report.files_deleted > 0 && `, ${repo.last_report.files_deleted} removed`}
            </div>
          )}
          {repo.error && <div className="form-error">{repo.error}</div>}
          <div className="repo-actions" onClick={(e) => e.stopPropagation()}>
            <button
              disabled={repo.state === "indexing"}
              onClick={() => onReindex(repo.id, false)}
              title="Index only files changed since the last indexed commit"
            >
              Sync
            </button>
            <button disabled={repo.state === "indexing"} onClick={() => onReindex(repo.id, true)}>
              Full re-index
            </button>
            <button className="danger" onClick={() => onDelete(repo.id)}>
              Remove
            </button>
          </div>
        </li>
      ))}
    </ul>
  );
}
