import { useState } from "react";

interface Props {
  text: string;
  onCorrect: (from: string, to: string) => void;
}

/// Renders text with quietly clickable words; clicking one opens a small
/// popover to teach unsound the intended word.
export function CorrectableText({ text, onCorrect }: Props) {
  const [edit, setEdit] = useState<{ word: string; value: string; x: number; y: number } | null>(
    null,
  );

  const tokens = text.split(/(\s+)/);

  const openFor = (token: string, e: React.MouseEvent) => {
    // Strip surrounding punctuation so "gently," corrects the word itself.
    const word = token.replace(/^[^\p{L}\p{N}'’-]+|[^\p{L}\p{N}'’-]+$/gu, "") || token;
    setEdit({ word, value: word, x: e.clientX, y: e.clientY });
  };

  const save = () => {
    if (!edit) return;
    const to = edit.value.trim();
    if (to && to !== edit.word) onCorrect(edit.word, to);
    setEdit(null);
  };

  return (
    <>
      {tokens.map((t, i) =>
        /\S/.test(t) ? (
          <span key={i} className="word" onClick={(e) => openFor(t, e)} title="Click to correct">
            {t}
          </span>
        ) : (
          t
        ),
      )}
      {edit && (
        <div
          className="correct-pop"
          style={{
            left: Math.min(edit.x, window.innerWidth - 280),
            top: Math.min(edit.y + 14, window.innerHeight - 90),
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <span className="correct-from">“{edit.word}” →</span>
          <input
            autoFocus
            value={edit.value}
            onChange={(e) => setEdit({ ...edit, value: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Enter") save();
              if (e.key === "Escape") setEdit(null);
            }}
            spellCheck={false}
          />
          <button className="quiet accent" onClick={save} title="Save correction">
            ✓
          </button>
          <button className="quiet" onClick={() => setEdit(null)} title="Cancel">
            ✕
          </button>
        </div>
      )}
    </>
  );
}
