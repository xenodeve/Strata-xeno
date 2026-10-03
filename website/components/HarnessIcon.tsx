import type { CSSProperties } from "react";

/**
 * The marks of the other coding apps whose skills, MCP servers and notes Strata can read (Import, Memory). The pictures are the
 * SVGs of LobeHub's static icon set (MIT), copied to /public/harness; each mark belongs to its maker and is here only to say which app
 * is meant. The shared `~/.agents` folder has no maker: its glyph is drawn here. A mark sits on a small light tile, so the dark ones
 * (Cursor, Antigravity) read on a dark page too.
 */
export type HarnessId = "claude-code" | "codex" | "antigravity" | "gemini-cli" | "cursor" | "claude-desktop" | "agents";

export const HARNESS_LABEL: Record<HarnessId, string> = {
  "claude-code": "Claude Code",
  codex: "Codex",
  antigravity: "Antigravity",
  "gemini-cli": "Gemini CLI",
  cursor: "Cursor",
  "claude-desktop": "Claude Desktop",
  agents: "~/.agents",
};

export function HarnessIcon({ id, size = 20, className = "" }: { id: HarnessId; size?: number; className?: string }) {
  const inner = Math.round(size * 0.72);
  return (
    <span className={"hicon " + className} style={{ width: size, height: size, "--hi": `${inner}px` } as CSSProperties} aria-hidden="true">
      {id === "agents" ? (
        <svg viewBox="0 0 24 24" width={inner} height={inner} fill="none" stroke="#4a4a52" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2.2h7a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2V7.5Z" />
          <path d="M12 11.2v4.2M9.9 13.3h4.2" />
        </svg>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={`/harness/${id}.svg`} alt="" width={inner} height={inner} loading="lazy" decoding="async" />
      )}
    </span>
  );
}
