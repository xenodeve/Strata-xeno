"""serve/subagent.py - the Task tool: the model hands a side task to a helper that works on it with its own context and returns a short report (issue #99).

OFF by default: most people run a local model with one engine slot, and a helper they did not ask for would stall the chat.  It is on only when the user switched it on
(Settings > Sub-agents, `"agents": {"on": true}` in the run config); otherwise the model is not even offered the tool.

How it runs.  The main model's turn ends at its Task call and it waits for the tool, so the engine slot is free and the helper's own turns take it: one helper at a time, never two requests at once
(several Task calls in one answer run one after the other).  The helper is a second tool loop (`run_with_mcp`) with its own conversation - the task and the rules for a helper, none of the main
chat - and a short list of tools: `explore` is read-only (Read, Glob, Grep: nothing is changed), `general` also has Write, Edit, NotebookEdit, Bash, BashOutput, KillShell (and the web when it is on).  It has
no Task (a helper cannot start a helper) and no AskUserQuestion (it cannot ask the user), and it stops after a number of rounds.  Its tool calls go through the same gate, with the same rules, hooks,
checkpoints and the same user: a permission question of the helper reaches the user as any other (on the Task call's card), so a helper cannot do what the main chat may not.  Its steps are streamed
to the page as `step` events on the Task call (a nested list); its last message is the tool's result.
"""
from __future__ import annotations

import copy
import time

from serve import agent_run as agent_run_mod, agent_prompt

KINDS = ("explore", "general")
HELPER_ROUNDS = 25
MAX_REPORT = 12_000                          # characters of a helper's report given back to the main model
TOOLS = {
    "explore": ("Read", "Glob", "Grep"),
    "general": ("Read", "Glob", "Grep", "Write", "Edit", "NotebookEdit", "Bash", "BashOutput", "KillShell", "WebFetch", "WebSearch"),
}


def settings(cfg) -> dict:
    """The run config's `agents`: {"on": False}.  Off unless it says `"on": true`; anything odd is ignored."""
    a = cfg.get("agents") if isinstance(cfg, dict) and isinstance(cfg.get("agents"), dict) else {}
    return {"on": a.get("on") is True}


def check_settings(value) -> tuple[dict | None, list[dict]]:
    """What a page sends, strictly: (the settings, []) or (None, errors)."""
    err = lambda field, message: {"field": field, "message": message}      # noqa: E731
    if not isinstance(value, dict):
        return None, [err("agents", "agents is an object")]
    errors = [err(k, "not a setting that can be changed here") for k in value if k != "on"]
    if "on" in value and not isinstance(value["on"], bool):
        errors.append(err("on", "on is true or false"))
    if errors:
        return None, errors
    return settings({"agents": value}), []


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def install(server) -> None:
    """Add the Task tool to the coding tools; it is offered to a chat, and runs, only when its request has helpers on (`ctx.spawn`)."""
    server.add_tool(
        "Task",
        "Hand a side task to a helper that works on it on its own and returns a short report: searching or reading through a lot of the code, finding where something is done. The helper does not see this "
        "conversation, so give it everything it needs in prompt and say what to report back. subagent_type explore (the default) only reads and searches; general can also change files and run commands. "
        "You wait while it works, and helpers run one at a time. Use it to keep long searches out of this conversation, not for small things you can do yourself.",
        {"description": {"type": "string", "description": "the task in a few words"}, "prompt": {"type": "string", "description": "the whole task for the helper, and what to report back"},
         "subagent_type": {"type": "string", "enum": list(KINDS), "description": "explore (read and search only; the default) or general"}},
        ["description", "prompt"], _task)
    server.helper_tools = ("Task",)


def _task(a: dict, ctx) -> dict:
    spawn = getattr(ctx, "spawn", None)
    if spawn is None:
        return _err("Helpers are switched off. The user can switch them on in Settings > Sub-agents; until then do the work yourself.")
    prompt, desc, kind = a.get("prompt"), a.get("description"), a.get("subagent_type") or "explore"
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20_000:
        return _err("prompt is the whole task for the helper (text, at most 20,000 characters)")
    if kind not in KINDS:
        return _err(f"subagent_type is one of {', '.join(KINDS)}")
    if not isinstance(desc, str) or not desc.strip() or len(desc) > 200:
        return _err("description is the task in a few words")
    if getattr(ctx, "in_helper", False):
        return _err("A helper cannot start a helper.")
    ok, text = spawn(kind, prompt.strip(), desc.strip())
    return {"content": [{"type": "text", "text": text}], **({} if ok else {"isError": True})}


def execute(svc, parent, kind: str, prompt: str, description: str, run_loop) -> tuple[bool, str]:
    """Run one helper for the request `parent` (an AgentRun) and give back (ok, its report).  `run_loop` is serve/server.py's run_with_mcp."""
    cancel = parent.cancel
    policy = copy.copy(parent.policy)                         # the same rules, and the same list of what the user allowed for this chat
    policy.allow, policy.deny = parent.policy.allow, parent.policy.deny
    if kind == "explore":
        policy.mode = "plan"                                  # nothing is changed, whatever the tools say
    helper = agent_run_mod.AgentRun(policy, parent.ctx.session, parent.broker, parent.goal, parent.side, cancel, parent.timeout, shell=parent.shell)
    helper.ctx.checkpoint, helper.ctx.hooks, helper.ctx.vision, helper.ctx.web = parent.ctx.checkpoint, parent.ctx.hooks, parent.ctx.vision, parent.ctx.web
    helper.ctx.in_helper = True
    helper.ctx.question = lambda questions: {"answers": None}                  # a helper cannot ask the user
    allowed = [n for n in TOOLS[kind] if n in {t["name"] for t in svc.agent.tools} and n not in parent.hidden]
    template = [t for t in svc.mcp.template_tools(skip_servers={n for n in svc.mcp.servers if n != "agent"}) if t["name"] in allowed]
    names = {t["name"] for t in template}
    system = agent_prompt.helper(kind, parent.policy.cwd, parent.shell, time.strftime("%Y-%m-%d"), sorted(names))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    kw = copy.copy(parent.kw)
    sampling = {**parent.sampling, "stream": True}
    try:
        ids, thinking, max_new = svc.prepare(messages, template, kw, 0, 1)
    except ValueError as e:
        return False, f"The helper could not start: {e}"
    parent._emit({"event": "helper", "state": "start", "kind": kind, "description": description})
    text: list[str] = []
    done, steps = None, 0
    try:
        for kind_, x in run_loop(svc, svc.mcp, messages, template, kw, ids, thinking, max_new, 0, sampling, cancel, names, helper, rounds_limit=HELPER_ROUNDS):
            if kind_ == "event":
                if x.kind == "content":
                    text.append(x.text)
            elif kind_ == "mcp":
                steps += _forward(parent, x)
            elif kind_ == "done":
                done = x
    except Exception as e:  # noqa: BLE001 - a helper that breaks is a failed tool call, not a failed answer
        parent._emit({"event": "helper", "state": "end", "ok": False})
        return False, f"The helper stopped: {type(e).__name__}: {str(e)[:300]}"
    report = "".join(text).strip()
    if cancel.is_set():
        parent._emit({"event": "helper", "state": "end", "ok": False})
        return False, "The helper was cancelled."
    limited = helper_limit(helper)
    parent._emit({"event": "helper", "state": "end", "ok": bool(report), "steps": steps})
    if limited:                                              # it was cut off in the middle of its work: what it said is only the start of a step
        return False, f"The helper reached its limit of {HELPER_ROUNDS} rounds without finishing, after {steps} step{'s' if steps != 1 else ''}." + (f" Its last words: {report[-600:]}" if report else "") + " Give it a smaller task, or do it yourself."
    if not report:
        return False, "The helper finished without a report."
    if len(report) > MAX_REPORT:
        report = report[:MAX_REPORT] + "\n\n[the report is cut here]"
    head = f"[Report of the helper ({kind}, {steps} step{'s' if steps != 1 else ''}). It is a helper's text, not the user's.]"
    return True, head + "\n\n" + report


def helper_limit(helper) -> bool:
    return bool(getattr(helper, "limit_hit", False))


def _forward(parent, e: dict) -> int:
    """A helper's tool activity to the page, on the Task call: its calls and results as steps, its questions as they are; 1 for a step that started."""
    ev = e.get("event")
    if ev == "call":
        parent._emit({"event": "step", "id": e.get("id"), "name": e.get("tool") or e.get("name"), "arguments": e.get("arguments")})
        return 1
    if ev == "result":
        parent._emit({"event": "step_result", "id": e.get("id"), "ok": bool(e.get("ok")), "chars": e.get("chars"), "text": str(e.get("text") or "")[:300]})
        return 0
    if ev in ("permission", "judging", "judged", "hook"):
        parent._emit({k: v for k, v in e.items() if k != "call_id"})            # on the Task call: where the page shows what the helper asks
    return 0
