import type { Message, ToolCall } from "./chat"

// A message is changed in place while it is written (the text grows, a tool's state changes), so the one object cannot say whether it is
// different from when it was last drawn. This is a short string that changes whenever anything the message draws changes: the page keeps
// the string from its last draw, and a message whose string is the same is not drawn again (a page that polls the server twice a second
// drew every message of a long conversation each time, and parsed the markdown of each).
// Strings are counted by length (a tool's result can be 20,000 characters, and this runs for every message at every draw).

const toolSig = (c: ToolCall): string => [
  c.id, c.state, c.open ? 1 : 0, c.ms, c.result?.length, c.chars, c.truncated ? 1 : 0, c.ok ? 1 : 0, c.judging ? 1 : 0, c.hookRunning, c.doneAt, c.at, c.rat, c.round,
  c.hooks?.length, c.helper && `${c.helper.state}:${c.helper.steps}`, c.steps && c.steps.map((s) => `${s.state}${s.text?.length ?? 0}`).join(","),
  c.ask && `${c.ask.answer}:${c.ask.stopped ? 1 : 0}:${c.ask.kept?.scope}`, c.question && `${c.question.stopped ? 1 : 0}:${JSON.stringify(c.question.answers ?? null)}`, c.judge && `${c.judge.verdict}${c.judge.severity}`,
].join("|")

export function messageSig(m: Message): string {
  return [
    m.role, m.time, m.text.length, m.reasoning?.length, m.thinkSecs, m.thinkAt, m.error, m.stopped ? 1 : 0, m.limit, m.hookRunning, m.meta,
    m.images?.length, m.files?.length, m.hooks?.length, m.stats && JSON.stringify(m.stats), m.compact && JSON.stringify(m.compact),
    m.prefill && JSON.stringify(m.prefill), m.todos && JSON.stringify(m.todos), m.tools?.map(toolSig).join(";"),
  ].join("~")
}
