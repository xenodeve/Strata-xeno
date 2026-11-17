# One Strata for a team: the central PC and the client PCs

Strata can serve a small team from one PC. This page explains which machine does what. It covers where a chat's
tools run, where its MCP servers run, and what the owner and a guest may each do. It is written from the code
(`serve/server.py`, `serve/runner.py`, `serve/runner_hub.py`, `serve/permissions.py`). Issues #310, #329, #338 and
#339 are the history.

## The two kinds of machine

- **The central PC** (the *Server PC* in the web app) runs the Strata server and the model on its graphics cards. Every
  prompt is read and every answer is written there, whoever sends it.
- **A client PC** (*My PC* in the web app) is a team member's own computer. Its browser opens the central PC's web app.
  Strata Desktop on it runs a small background program, the *runner*, which connects to the server with the member's key.
  The runner is what lets the chat work on that PC's own files.

Without Strata Desktop, a client PC is only a browser: everything a chat does happens on the central PC.

## A project on My PC, or on the server

When you start a chat with a project folder, the web app asks whether the folder is on **My PC** or on the
**Server PC**. That choice decides where the coding tools run.

| | Project on My PC (through the runner) | Project on the Server PC |
|---|---|---|
| Read, Write, Edit, Glob, Grep, Bash, TodoWrite | on your PC, in your folder | on the central PC, in its folder |
| The files you mention with `@` | on your PC | on the central PC |
| WebFetch, WebSearch, Task (helpers) | on the central PC | on the central PC |
| StrataDocs, StrataSettings, StrataSettingsChange | on the central PC | on the central PC |
| The model itself | on the central PC | on the central PC |

So in a My PC chat, what the tools read on your PC goes to the central PC as part of the conversation. In a Server PC
chat, a command's `localhost` is the central PC: your computer only when you are working on the central PC itself.

## MCP servers: on My PC or on the central PC

There are two sets of MCP servers:

- **The central PC's**, set up in **Settings > MCP servers** (the run config). They run on the central PC. The
  model sees their tools as `<server>__<tool>`.
- **My PC's**, from the runner's own file (`%LOCALAPPDATA%\StrataRunner\runner.json`; today you edit that file by hand, and an editor in the web app is planned in #338). They run on
  your PC. The model sees their tools with the prefix `my_`, as `my_<server>__<tool>`, and the skills of your PC as
  `my_skills`.

The `my_` prefix is the whole rule, and the chat's rules tell the model about it:

- `localhost` in a `my_` tool's call is your machine. In any other MCP tool it is the central PC.
- A browser tool (Playwright, for example) that must open a site you are developing on your PC has to be the `my_` one.
  The central PC's copy cannot reach your PC's `localhost`.
- A file path in a `my_` tool is a path on your PC. In any other tool it is a path on the central PC.

## The owner and the guests

A team server gives out named keys, each with a role (Settings > Keys and devices):

- **The owner** may change the central PC's settings: the run config, the MCP servers, web access, helpers, hooks
  and keys. The owner may also use the coding tools on the central PC's own folders.
- **A guest** uses the model and their own PC through the runner. A guest cannot change the central PC's settings,
  and cannot run the coding tools in the central PC's folders. A guest sees only the central PC's MCP servers the owner
  marked for guests (`"guest_mcp"` in the run config), the imported skills, and Strata's own docs (StrataDocs).
  StrataSettings and StrataSettingsChange are not offered to a guest, and refuse one.
- A guest's MCP servers are those of their own PC, set in its runner's file. The owner cannot change a guest's PC. (#338
  plans an editor for them in the web app, for the PC's own user only.)
- A guest's WebFetch reaches only the public internet: never the central PC, a private network, Tailscale or Radmin.

## YOLO mode on each machine

YOLO mode (#329) runs the coding tools' calls without asking.

- **The owner** may use YOLO in a Server PC project and in a My PC project.
- **A guest** gets YOLO only in a My PC project, where the calls run on the guest's own computer. (A guest has no
  coding tools on the central PC at all.)
- In every case, YOLO still refuses commands that can do harm that is hard to undo, and anything that would send a
  secret off the PC (an `.env` file, a key, a git push of commits that hold one).
- A change of Strata's own settings (StrataSettingsChange) asks the user in every mode, YOLO included. Auto mode's
  judge never decides it.

## What the chat's model may change in Strata

When you ask the model to change Strata itself, it uses StrataSettingsChange: one setting per call, and a card with the
value now and the new value, which you allow or refuse. It may change only settings that take effect without a restart:

- web access and its search provider;
- helpers;
- the central PC's MCP servers (add, edit, switch off, delete, import from another app);
- the skill import switches;
- the Chat's shared defaults (thinking level, sampling).

It cannot change the engine's flags (context, parallel requests, cache slots), keys, hooks, the API key, the allowed
hosts or origins, or the Brave key. You change those in Settings or in the run config.
