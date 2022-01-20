export interface ParsedCitation {
  path: string;
  start: number;
  end: number;
}

export type AnswerPart =
  | { kind: "text"; text: string }
  | { kind: "citations"; items: ParsedCitation[]; raw: string };

const BRACKET = /\[([^[\]\n]{3,400})\](?!\()/g;
const ONE = /^`?([\w.@+\-/]+):L?(\d+)(?:\s*[-–]\s*L?(\d+))?`?$/;

/** Split answer text into prose and citation groups, mirroring the server parser. */
export function splitAnswer(text: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let last = 0;
  for (const match of text.matchAll(BRACKET)) {
    const items: ParsedCitation[] = [];
    for (const piece of match[1].split(/[;,]/)) {
      const m = ONE.exec(piece.trim());
      if (!m) continue;
      const a = Number(m[2]);
      const b = m[3] ? Number(m[3]) : a;
      items.push({ path: m[1].replace(/^\.\//, ""), start: Math.min(a, b), end: Math.max(a, b) });
    }
    if (items.length === 0) continue;
    const index = match.index ?? 0;
    if (index > last) parts.push({ kind: "text", text: text.slice(last, index) });
    parts.push({ kind: "citations", items, raw: match[0] });
    last = index + match[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
  return parts;
}

export function citationKey(c: ParsedCitation): string {
  return `${c.path}:${c.start}-${c.end}`;
}

export function citationLabel(c: ParsedCitation): string {
  return c.start === c.end ? `${c.path}:${c.start}` : `${c.path}:${c.start}-${c.end}`;
}
