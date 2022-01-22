import { citationLabel, type ParsedCitation } from "../citations";

interface Props {
  citation: ParsedCitation;
  verified: boolean;
  active: boolean;
  onOpen: (c: ParsedCitation) => void;
}

export function CitationChip({ citation, verified, active, onOpen }: Props) {
  const classes = ["chip", verified ? "chip-ok" : "chip-warn", active ? "chip-active" : ""];
  return (
    <button
      type="button"
      className={classes.join(" ").trim()}
      title={verified ? "Open cited lines" : "Not found in retrieved context"}
      onClick={() => onOpen(citation)}
    >
      <span className="chip-file">{citation.path.split("/").pop()}</span>
      <span className="chip-lines">
        :{citation.start}
        {citation.end !== citation.start && `-${citation.end}`}
      </span>
      <span className="sr-only">{citationLabel(citation)}</span>
    </button>
  );
}
