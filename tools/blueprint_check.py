"""The commit-time blueprint check (xeno, 2026-09-30): a commit that changes what docs/BLUEPRINT.md describes must
update it in the same commit, or say why not in a trailer:

    Blueprint: n/a - <why this change needs no blueprint edit>

It looks at the staged diff for the changes AGENTS.md lists that can be told from a diff line: an engine flag, a GEN
key, a STRATA_* environment variable, a run-config key, a new source file. It cannot see a changed request path or a
moved thread, so it is a floor, not the whole rule.

Run by tools/hooks/commit-msg (install once per clone: `git config core.hooksPath tools/hooks`), or by hand:
    python tools/blueprint_check.py <commit message file>
"""
import re
import subprocess
import sys

BLUEPRINT = "docs/BLUEPRINT.md"
TRAILER = re.compile(r"(?mi)^Blueprint:\s*(updated\b|n/?a\s*[-:]\s*\S.*)")
SOURCE = re.compile(r"^(src|include|serve|setup\.py$|CMakeLists\.txt$)")
NOT_SOURCE = re.compile(r"(^|/)(test_[^/]*|[^/]*_test\.[^/]*|[^/]*_parity\.[^/]*)$|^(tests|docs)/")
PATTERNS = [
    (re.compile(r'\ba\s*==\s*"(--[\w-]+)"'), "engine flag {}"),
    (re.compile(r'\bkey\s*==\s*"(\w+)"'), "GEN key {}"),
    (re.compile(r'getenv\(\s*"(STRATA_\w+)"'), "environment variable {}"),
    (re.compile(r'os\.environ(?:\.get)?[\(\[]\s*"(STRATA_\w+)"'), "environment variable {}"),
    (re.compile(r'\bcfg(?:\.get\(|\[)\s*"(\w+)"'), "run-config key {}"),
]


def _files(diff):
    """(path, added?, is_new_file, [changed lines]) per file of a unified diff."""
    out, cur = [], None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            cur = [line.split(" b/", 1)[-1], False, []]
            out.append(cur)
        elif cur is None:
            continue
        elif line.startswith("new file mode"):
            cur[1] = True
        elif (line.startswith("+") and not line.startswith("+++")) or (line.startswith("-") and not line.startswith("---")):
            cur[2].append(line)
    return out


def reasons(diff):
    """What in the diff changes something the blueprint describes (empty: nothing it can see)."""
    found = []
    for path, new, lines in _files(diff):
        if not SOURCE.match(path) or NOT_SOURCE.search(path):
            continue
        if new:
            found.append(f"new source file {path}")
        seen = {"+": [], "-": []}     # an item on both sides is an edited line, not a change (net difference only)
        for line in lines:
            code = line[1:].split("//")[0] if path.endswith((".cpp", ".cu", ".hpp", ".h")) else line[1:].split("#")[0]
            for rx, what in PATTERNS:
                for m in rx.finditer(code):
                    item = what.format(m.group(1))
                    if item not in seen[line[0]]:
                        seen[line[0]].append(item)
        for sign, verb in (("+", "added"), ("-", "removed")):
            other = seen["-" if sign == "+" else "+"]
            found += [f"{item} {verb} in {path}" for item in seen[sign] if item not in other]
    return found


def check(diff, staged, message):
    """Error lines for this commit; empty when it may go ahead."""
    why = reasons(diff)
    if not why or BLUEPRINT in staged or TRAILER.search(message or ""):
        return []
    return ["docs/BLUEPRINT.md is not in this commit, but the commit changes what it describes:"] + \
        [f"  - {r}" for r in why] + \
        ["Update docs/BLUEPRINT.md in this commit (AGENTS.md, 'Keep docs/BLUEPRINT.md current'), or add a trailer",
         "  Blueprint: n/a - <why this change needs no blueprint edit>"]


def main(argv):
    message = open(argv[1], encoding="utf-8", errors="replace").read() if len(argv) > 1 else ""
    diff = subprocess.run(["git", "diff", "--cached", "-U0", "--no-color"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True).stdout.split()
    errors = check(diff, staged, message)
    for line in errors:
        print(line, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
