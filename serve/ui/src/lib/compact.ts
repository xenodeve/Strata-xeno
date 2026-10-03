// Compacting a conversation (issue #96), as Claude Code's /compact does: when the context is nearly full - or when the user says so - the model is asked to write a summary
// of everything so far, and the summary takes the place of the messages. The conversation goes on from the summary. The model is called as for any answer, with no tools.
// This file is the text of that request and the numbers that decide when; the controller (lib/chat.ts) does the calling.

export const COMPACT_AT = 0.95             // the share of the context at which a conversation is compacted by itself, before the next prompt is sent (the rest is room for the summary)
export const MAX_SUMMARY = 4096            // the most tokens a summary may take
export const MIN_SUMMARY = 512             // less room than this and the oldest messages are left out of what is summarised

/** Tokens in a text, guessed from its length (about 2.6 characters a token: on the safe side for code and for Thai). The exact figure is what the last answer reported. */
export const estimateTokens = (x: unknown): number => Math.ceil((typeof x === "string" ? x : JSON.stringify(x) ?? "").length / 2.6)

/** The request for the summary: the sections Claude Code's own summary has, and what to leave out. `focus` is what the user typed after `/compact`. */
export function compactPrompt(focus = ""): string {
  const extra = focus.trim() ? `\n\nAdditional instructions from the user for this summary:\n${focus.trim()}` : ""
  return [
    "Your task is to write a detailed summary of the conversation so far, so that the work can go on from the summary alone. Be thorough about what the user asked for and what was done.",
    "Respond with TEXT ONLY. Do not call any tools: you have all you need in the conversation above.",
    "",
    "First write an <analysis> in which you go through the conversation in order and note what matters (keep it short). Then write the <summary> with these sections:",
    "1. Primary request and intent: what the user asked for, in detail.",
    "2. Key concepts: the technologies, decisions and rules that matter.",
    "3. Files and code: the files that were looked at, changed or made, with the important snippets and why each matters.",
    "4. Errors and fixes: what went wrong and how it was fixed, including what the user said about it.",
    "5. Problem solving: what was solved and what is still being worked on.",
    "6. All user messages: every message of the user that is not a tool result, nearly word for word. They show the intent and the changes of direction.",
    "7. Pending tasks: what was asked and is not done yet.",
    "8. Current work: exactly what was being done just before this summary, with file names and snippets.",
    "9. Next step: the step that follows directly from the current work and from the user's latest request, quoting that request. Do not start work the user did not ask for.",
    "",
    "Format: <analysis>...</analysis> then <summary>...</summary>. Write in the language the user writes in." + extra,
  ].join("\n")
}

/** The summary in what the model wrote: what is inside <summary>, else (the model left the tags out) the text without its <analysis>. Empty when there is nothing. */
export function summaryOf(raw: string): string {
  const closed = /<summary>([\s\S]*?)<\/summary>/i.exec(raw)
  if (closed) return closed[1].trim()
  const open = /<summary>([\s\S]*)$/i.exec(raw)                                 // cut off by the limit on its length: what there is of it
  if (open) return open[1].trim()
  return raw.replace(/<analysis>[\s\S]*?(<\/analysis>|$)/i, "").trim()
}

/** What the conversation says in place of the messages that were summarised: the model reads it as the earlier part of the conversation. */
export function continuationText(summary: string): string {
  return [
    "This conversation is continuing from an earlier one that grew past the limit of the context. The summary below covers the earlier part.",
    "",
    summary,
    "",
    "Continue from where the conversation left off. Do not ask the user anything that the summary already answers, and do not start work that was not asked for.",
  ].join("\n")
}

/** Whether a conversation that uses `used` tokens of a context of `max`, with a prompt of about `incoming` tokens to come, should be compacted first. */
export const shouldCompact = (used: number, incoming: number, max: number): boolean => max > 0 && used + incoming >= max * COMPACT_AT

/** The `/compact` command: the words after it (what to focus on), or null when the text is not that command. A skill by another name does not match. */
export function compactCommand(text: string): { focus: string } | null {
  const m = /^\s*\/compact(?:\s+([\s\S]*))?$/i.exec(text)
  return m ? { focus: (m[1] ?? "").trim() } : null
}

/** How many tokens the summary may take, from what is left of the context after the conversation and the request; below MIN_SUMMARY it is too little. `history` is what the
 *  conversation uses: the figure the server reported when there is one (the safe one: a guess from the text is too low for Thai), else a guess. */
export const summaryRoom = (max: number, history: number, request: number): number => (max > 0 ? Math.min(MAX_SUMMARY, max - history - request - 24) : MAX_SUMMARY)
