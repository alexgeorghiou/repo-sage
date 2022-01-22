import { useState, type FormEvent } from "react";

interface Props {
  onAdd: (source: string, name?: string, branch?: string) => Promise<void>;
}

export function AddRepoForm({ onAdd }: Props) {
  const [source, setSource] = useState("");
  const [branch, setBranch] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!source.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await onAdd(source.trim(), undefined, branch.trim() || undefined);
      setSource("");
      setBranch("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="add-repo" onSubmit={submit}>
      <input
        aria-label="Repository URL or path"
        placeholder="https://github.com/org/repo.git"
        value={source}
        onChange={(e) => setSource(e.target.value)}
      />
      <div className="add-repo-row">
        <input
          aria-label="Branch"
          placeholder="branch (optional)"
          value={branch}
          onChange={(e) => setBranch(e.target.value)}
        />
        <button type="submit" disabled={busy || !source.trim()}>
          {busy ? "Adding…" : "Add"}
        </button>
      </div>
      {error && <p className="form-error">{error}</p>}
    </form>
  );
}
