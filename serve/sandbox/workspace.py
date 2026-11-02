"""serve/sandbox/workspace.py - a copy of a project that a model can change and test, and the way its changes are offered back (design: docs/plans/2026-11-02-sandbox-workspace-design.md, 4.4 and 4.7).

The copy is the working tree as it is now, not HEAD: tracked files and untracked ones that are not ignored, byte for byte and never hardlinked. `.git`, whatever `permissions.is_secret` catches,
files over the size limit, build output and anything reached through a link stay out, and `excluded` says why. The bytes each file had at the snapshot are kept in one compressed file outside
the copy, so that a diff still works once the real project has moved on.

This module never writes to the real project: `apply_plan` says what should be written, and the caller does it through the existing permission and checkpoint path. A file changed on both sides is shown by `conflict`
and settled by `resolve`, which reads the real file and changes only the copy and the base.
It runs with the server's rights while code in the box may do anything to the copy, so nothing here follows a link it finds there, and a path from the copy is checked again before it may
name a path in the project."""
from __future__ import annotations

import difflib
import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from serve import files, gitview, permissions

MAX_FILE = 2 * 1024 * 1024             # a file bigger than this is left out of the copy
QUOTA = 2 * 1024 * 1024 * 1024         # bytes one workspace may hold on disk: the copy, the kept base and the saved output
RESERVED = ".strata-sandbox"           # a folder of the copy for the sandbox's own files: never part of a change, never offered to the project
GIT_TIMEOUT = 30.0                     # seconds one git command may take
KEEP_RECORDS = 50                      # test records kept for one workspace: the newest
MAX_DIFF_SPAN = 2000                   # lines from the first change to the last that difflib is given. Its worst measured case here is 1 s; at 4,000 it was 8 s, and a reformat of every second line of 1 MB took 53 s
WORKERS =min(8, os.cpu_count() or 1)  # the first open of a file just written waits about 4 ms (measured here, an on-access scan by the look of it); it runs in the opener's thread, so workers divide it: 16 s by one, 2.4 s by 8
MTIME_TICK_NS = 100_000_000            # a hash is reused by size and mtime only for a file older than this when hashed: a second write of the same size inside one tick of the file clock (about 16 ms on NTFS) would keep both
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
LINK_TAG = 0x20000000                  # the "name surrogate" bit of a reparse tag: a symlink, a junction. A cloud placeholder has a tag too, and is an ordinary file
GIT_REPO_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR", "GIT_PREFIX")     # set when the server was started from a git hook: they would point git at another repository

_DEVICE = re.compile(r"(con|prn|aux|nul|conin\$|conout\$|com\d|lpt\d)(\..*)?\Z", re.I)
_GIT_NAME = re.compile(r"\.git\Z|git~\d+\Z", re.I)                            # git~1 is the short name Windows gives .git


SOURCE = {"py", "pyi", "js", "jsx", "ts", "tsx", "go", "rs", "java", "kt", "c", "cc", "cpp", "h", "hpp", "cs", "rb", "php", "swift"}


class WorkspaceError(Exception):
    pass


class QuotaExceeded(WorkspaceError):
    pass


def private(rel: str) -> bool:
    """What stays out of the copy. Source code stays in whatever its name: `src/secrets.py` out of the copy fails the project's imports for an unrelated reason, and a
    box has no network for it to leave by."""
    return permissions.is_secret(rel) and rel.rsplit(".", 1)[-1].lower() not in SOURCE


class ApplyRefused(WorkspaceError):
    """The delta cannot go to the project. `reasons`: [{"path", "why"}], every problem found, so that one answer settles them all."""

    def __init__(self, reasons: list[dict]):
        super().__init__("; ".join(f"{r['path']}: {r['why']}" for r in reasons[:5]))
        self.reasons = reasons


class ConflictMoved(WorkspaceError):
    """The real file is no longer the one the decision was made on: a change arrived while the conflict was being settled."""


@dataclass(frozen=True)
class Kept:
    """What a file was at the base: its hash and size, and where its bytes are in the kept store."""
    hash: str
    size: int
    offset: int
    length: int


@dataclass(frozen=True)
class Write:
    """One step for the caller. `base`: the hash the project's file must still have (None: there must be no file), to be checked again right before the step."""
    path: str                              # relative, with /
    full: str                              # the path in the real project
    action: str                            # "write" | "delete"
    data: bytes | None
    base: str | None


@dataclass(frozen=True)
class Plan:
    writes: list[Write]
    delta: str                             # the hash of the delta these writes carry
    tested: dict | None                    # the newest `tested_record` of exactly this delta, passed or not; None: this delta was not tested


# ------------------------------------------------------------------------------------------------ paths
def _ext(path: str) -> str:
    """A path Windows takes at any length: the copy sits deeper than the project, so a name that fits there may not fit here."""
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    return "\\\\?\\" + path.replace("/", "\\")


def _plain(path: str) -> str:
    return path[4:] if path.startswith("\\\\?\\") else path


def _real(path: str) -> str:
    return _plain(os.path.realpath(_ext(path)))


def _same(a: str, b: str) -> bool:
    return os.path.normcase(a) == os.path.normcase(b)


def _join(root: str, rel: str) -> str:
    return os.path.join(root, *rel.split("/")) if rel else root


def _lstat(path: str) -> os.stat_result | None:
    try:
        return os.lstat(_ext(path))
    except (OSError, ValueError):
        return None


def _is_link(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_reparse_tag", 0) & LINK_TAG)


def _local_folder(path, what: str) -> str:
    raw = _plain(os.fspath(path))
    if os.name == "nt" and raw.startswith(("\\\\", "//")):
        raise WorkspaceError(f"{what} is on a network path, which this version does not take")
    real = os.path.realpath(os.path.abspath(raw))
    if os.name == "nt" and real.startswith("\\\\"):                           # a mapped network drive resolves to one
        raise WorkspaceError(f"{what} is on a network path, which this version does not take")
    return real


def bad_path(rel: str) -> str | None:
    """Why a path offered by the copy cannot be a path of the project, or None."""
    if not isinstance(rel, str) or not rel or "\0" in rel or "\\" in rel or rel.startswith("/") or (os.name == "nt" and ":" in rel):
        return "it is not a path inside the project"
    for part in rel.split("/"):
        if part in ("", ".", ".."):
            return "it is not a path inside the project"
        if _GIT_NAME.match(part.rstrip(" .")):
            return "it is inside .git"
        if os.name == "nt" and (part != part.rstrip(" .") or _DEVICE.match(part)):
            return "Windows would read this name as another file"
    return None


def _is_output(rel: str) -> bool:
    return any(p in files.SKIP_DIRS or p.startswith("build-") for p in rel.split("/")[:-1])


def _lines(data: bytes) -> list[str]:
    """Lines by "\\n" alone, as git has them: str.splitlines also breaks on form feed and U+2028, and would show changes that are not there."""
    return [x for x in re.split(r"(?<=\n)", data.decode("utf-8", "replace")) if x]


def _changed_span(a: list[str], b: list[str]) -> int:
    """Lines between the first difference and the last, on the longer side. It is this and not the size of the file that sets what difflib costs."""
    shorter = min(len(a), len(b))
    head = 0
    while head < shorter and a[head] == b[head]:
        head += 1
    tail = 0
    while tail < shorter - head and a[-1 - tail] == b[-1 - tail]:
        tail += 1
    return max(len(a), len(b)) - head - tail


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        block = memoryview(bytearray(1 << 16))                                # one buffer: a fresh 1 MiB for each read took 1.6 s over a tree whose bytes hash in 0.24 s
        with open(_ext(path), "rb") as f:
            while n := f.readinto(block):
                h.update(block[:n])
        return h.hexdigest()
    except OSError:
        return None


def _git(folder: str, *args: str, stdin: bytes | None = None) -> tuple[int, bytes, str]:
    """(exit code, what it wrote, what it complained about). The flags and the environment are gitview's; this one also takes input, and answers beyond gitview's limit on output."""
    exe = gitview.available()
    if not exe:
        raise WorkspaceError("git is not installed")
    env = gitview._env()
    for key in GIT_REPO_ENV:
        env.pop(key, None)
    feed = {"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}
    try:
        p = subprocess.run([exe, *gitview.SAFE, "-c", "core.longpaths=true", *args], cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=GIT_TIMEOUT, creationflags=NO_WINDOW, **feed)
    except (OSError, subprocess.SubprocessError) as e:
        raise WorkspaceError(f"git could not run: {e}") from e
    return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace").strip()


def _force(func, path, _exc) -> None:
    """For rmtree: what stopped it is usually a read-only mark, which Windows honours for deleting too."""
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        func(path)
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------ the workspace
class Workspace:
    def __init__(self, project_dir, scratch_root, max_file: int = MAX_FILE, quota: int = QUOTA):
        self.max_file, self.quota = max_file, quota
        self.project = _local_folder(project_dir, "the project")
        scratch = _local_folder(scratch_root, "the scratch folder")
        if not os.path.isdir(self.project):
            raise WorkspaceError("the project folder does not exist")
        if permissions.inside(scratch, self.project) or permissions.inside(self.project, scratch):
            raise WorkspaceError("the scratch folder and the project must be apart")
        code, out, err = _git(self.project, "rev-parse", "--is-inside-work-tree", "--show-prefix")
        lines = out.split(b"\n")
        if code != 0 or lines[0] != b"true":
            raise WorkspaceError(f"the project is not a git work tree: {err or lines[0].decode('utf-8', 'replace')}")
        self._prefix = os.fsdecode(lines[1]) if len(lines) > 1 else ""        # `git status` names paths from the top of the repository even when asked from a sub-folder

        self.base: dict[str, Kept] = {}
        self.excluded: dict[str, str] = {}                                    # path -> why it is not in the copy
        self.tested: list[dict] = []
        self.conflicts: set[str] = set()                                      # what the last sync found changed on both sides; `conflict` is what checks it now
        self.closed = False
        self._dirty: set[str] = set()                                         # what `git status` listed last: a file listed then and clean now has moved too, and status alone would not say so
        self._parents: dict[str, bool] = {}
        self._ignored: dict[str, bool] = {}
        self._repos: set[str] = set()                                         # submodules and nested repositories, with a trailing /
        self._hashes: dict[str, tuple[tuple[int, int], str]] = {}             # path in the copy -> (size, mtime_ns) and the hash they had
        self._used: int | None = None

        os.makedirs(scratch, exist_ok=True)
        self.root = tempfile.mkdtemp(prefix="ws-", dir=scratch)
        self.copy = os.path.join(self.root, "copy")
        self._pack = os.path.join(self.root, "base.pack")                     # outside the copy, so the box has no way to change what a change is measured against
        try:
            self._snapshot()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _check_open(self) -> None:
        # without this a copy that has gone away reads as "every file was deleted", and that would be offered to the project
        if self.closed or not os.path.isdir(self.copy):
            raise WorkspaceError("this workspace is closed or its copy has gone")

    # -------------------------------------------------------------------------------------------- what is in the copy
    def _through_link(self, rel: str) -> bool:
        """Whether a folder on the way to `rel` is a link. git lists files through a junction, so its word on this is not enough."""
        parent = rel.rpartition("/")[0]
        if not parent:
            return False
        hit = self._parents.get(parent)
        if hit is None:
            want = _join(self.project, parent)
            hit = self._parents[parent] = not _same(_real(want), want)
        return hit

    def _left_out(self, rel: str, tracked: bool, st: os.stat_result) -> str | None:
        """Why a file of the project does not go into the copy, or None."""
        why = bad_path(rel)
        if why:
            return why
        if self.is_output(rel):
            return "the sandbox keeps this folder for itself"
        if private(rel):
            return "it looks like a secret"
        if not tracked and _is_output(rel):
            return "it is build output"
        if self._through_link(rel) or _is_link(st):
            return "it is a link or is reached through one"
        if not stat.S_ISREG(st.st_mode):
            return "it is not a regular file"
        if st.st_size > self.max_file:
            return f"it is over the size limit ({st.st_size:,} bytes)"
        return None

    def _walk_git(self, prefix: str = ""):
        """(path, tracked, stat) for what git lists in the project now, also inside submodules and nested repositories."""
        code, out, err = _git(_join(self.project, prefix.rstrip("/")), "ls-files", "-z", "-t", "-c", "-o", "--exclude-standard")
        if code != 0:
            raise WorkspaceError(f"git could not list the project: {err}")
        seen: set[bytes] = set()
        for item in out.split(b"\0"):
            if len(item) < 3 or item in seen:
                continue
            seen.add(item)
            rel = prefix + os.fsdecode(item[2:]).rstrip("/")
            st = _lstat(_join(self.project, rel))
            if st is None:                                                    # tracked, and deleted in the working tree
                continue
            if stat.S_ISDIR(st.st_mode) and not _is_link(st):
                if not self._through_link(rel) and _lstat(_join(self.project, rel + "/.git")) is not None:
                    self._repos.add(rel + "/")
                    yield from self._walk_git(rel + "/")
                continue
            yield rel, not item.startswith(b"?"), st

    def _status(self) -> tuple[str | None, dict[str, bool]]:
        """HEAD, and what `git status` lists now: path relative to the project -> tracked. One git run: this happens before every command in the box."""
        code, out, err = _git(self.project, "status", "--porcelain=v2", "--branch", "-z", "--untracked-files=all", "--no-renames", "--ignore-submodules=none", "--", ".")
        if code != 0:
            raise WorkspaceError(f"git status failed: {err}")
        st = gitview._parse_status(os.fsdecode(out))
        found: dict[str, bool] = {}
        for group in ("staged", "unstaged", "untracked", "conflicted"):
            for item in st[group]:
                rel = item["path"]
                found[rel[len(self._prefix):] if rel.startswith(self._prefix) else rel] = group != "untracked"
        return st["head"]["oid"], found

    def _read_real(self, rel: str) -> bytes | None:
        try:
            with open(_ext(_join(self.project, rel)), "rb") as f:
                data = f.read(self.max_file + 1)
        except OSError:
            return None
        return data if len(data) <= self.max_file else None

    def _keep(self, pack, digest: str, size: int, packed: bytes) -> Kept:
        offset = pack.seek(0, os.SEEK_END)
        pack.write(packed)
        return Kept(digest, size, offset, len(packed))

    def _put(self, rel: str, data: bytes, mode: int | None = None) -> None:
        full = _join(self.copy, rel)
        os.makedirs(_ext(os.path.dirname(full)), exist_ok=True)
        tmp = f"{full}.strata-{os.getpid()}"
        with open(_ext(tmp), "wb") as f:
            f.write(data)
        if mode is not None and os.name != "nt":
            os.chmod(tmp, (mode & 0o777) | 0o200)
        os.replace(_ext(tmp), _ext(full))

    def _copy_one(self, item: tuple[str, os.stat_result]) -> str | tuple[str, int, bytes]:
        """Copies one file, from one read of it, so that the hash and the kept bytes are of what the copy holds. (hash, size, compressed bytes), or why it stays out."""
        rel, st = item
        data = self._read_real(rel)
        if data is None:
            return "it could not be read"
        full = _join(self.copy, rel)
        try:
            with open(_ext(full), "xb") as f:                                 # "x": two names that differ only by case are one file here, and the second must not replace the first
                f.write(data)
        except FileExistsError:
            return "its name differs from another file's only by case, which this file system cannot hold"
        if os.name != "nt":
            os.chmod(full, (st.st_mode & 0o777) | 0o200)                      # a script must stay runnable; on Windows the only mode is the read-only mark, which would stop the model's edit
        return _hash(data), len(data), zlib.compress(data, 1)

    def _snapshot(self) -> None:
        self.head, status = self._status()                                    # before the first read: a change during the copy shows at the next sync, not never
        self._dirty = set(status)
        keep = []
        for rel, tracked, st in self._walk_git():
            why = self._left_out(rel, tracked, st)
            if why:
                self.excluded[rel] = why
            else:
                keep.append((rel, st))
        total = sum(st.st_size for _, st in keep)
        if total > self.quota:
            raise QuotaExceeded(f"the project needs {total:,} bytes and the quota is {self.quota:,}")
        for folder in {os.path.dirname(_join(self.copy, rel)) for rel, _ in keep} | {self.copy}:
            os.makedirs(_ext(folder), exist_ok=True)
        used = 0
        with open(self._pack, "ab+") as pack, ThreadPoolExecutor(WORKERS) as pool:
            for (rel, _), done in zip(keep, pool.map(self._copy_one, keep)):
                if isinstance(done, str):
                    self.excluded[rel] = done
                    continue
                self.base[rel] = kept = self._keep(pack, *done)
                used += kept.size + kept.length
                if used > self.quota:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise QuotaExceeded(f"the copy passed the quota of {self.quota:,} bytes")
        found, _ = self._walk_copy()                                          # the hash of each file is known without opening it, except for the last ones written, which the box could write again within the same tick
        settled = time.time_ns() - MTIME_TICK_NS
        self._hashes = {rel: (found[rel], kept.hash) for rel, kept in self.base.items() if rel in found and found[rel][1] < settled}

    # -------------------------------------------------------------------------------------------- reading and writing the copy
    def _checked(self, rel: str) -> str:
        """The path of `rel` in the copy, refusing what is not a path there or goes through a link the box planted."""
        why = bad_path(rel)
        if why:
            raise WorkspaceError(f"{rel}: {why}")
        full = _join(self.copy, rel)
        if not _same(_real(full), full):
            raise WorkspaceError(f"{rel}: it goes through a link")
        return full

    def _in_real_folder(self, rel: str) -> str:
        """The path of `rel` in the copy, refusing a link on the way to it. Unlike `_checked` the last part may itself be a link, for the code that replaces or removes the link and does not go through it."""
        parent = rel.rpartition("/")[0]
        if parent:
            self._checked(parent)
        return _join(self.copy, rel)

    def read_bytes(self, rel: str, limit: int = -1) -> bytes:
        self._check_open()
        full = self._checked(rel)
        if private(rel):
            raise WorkspaceError(f"{rel}: it looks like a secret")
        with open(_ext(full), "rb") as f:
            return f.read(limit)

    def size_of(self, rel: str) -> int:
        return os.lstat(_ext(self._checked(rel))).st_size

    @staticmethod
    def is_output(rel: str) -> bool:
        """Whether a path is in the sandbox's own folder of the copy."""
        return rel.split("/")[0].lower() == RESERVED

    def read_stamped(self, rel: str, stamp: tuple[int, int]) -> bytes:
        """The bytes of a file `listing` has just stamped. One the copy still holds unchanged comes from the kept store, which is one file to open: a first search would otherwise open every file of a fresh copy, at 16 s on this repository."""
        known = self._hashes.get(rel)
        kept = self.base.get(rel)
        if known and kept and known[0] == stamp and known[1] == kept.hash:
            return self._base_bytes(rel)
        return self.read_bytes(rel)

    def write_bytes(self, rel: str, data: bytes) -> None:
        """Changes a file of the copy, within the quota. What the box writes itself is held by the box's own limits; this is for the code that runs out here."""
        self._check_open()
        st = _lstat(self._checked(rel))
        now = self.usage() - (st.st_size if st else 0)
        if now + len(data) > self.quota:
            raise QuotaExceeded(f"{rel}: {now + len(data):,} bytes would pass the quota of {self.quota:,}")
        self._put(rel, data)

    def delete(self, rel: str) -> None:
        """Takes a file out of the copy, or a link the box left: the link is what goes, never what it points at."""
        self._check_open()
        why = bad_path(rel)
        if why:
            raise WorkspaceError(f"{rel}: {why}")
        full = self._in_real_folder(rel)
        st = _lstat(full)
        if st is None:
            raise FileNotFoundError(rel)
        if stat.S_ISDIR(st.st_mode) and not _is_link(st):
            raise WorkspaceError(f"{rel}: it is a folder")
        os.unlink(_ext(full))

    def rename(self, rel: str, new: str) -> None:
        self._check_open()
        old_full, new_full = self._checked(rel), self._checked(new)
        if _lstat(old_full) is None:
            raise FileNotFoundError(rel)
        if _lstat(new_full) is not None and not _same(old_full, new_full):
            raise WorkspaceError(f"{new}: there is a file there already")
        os.makedirs(_ext(os.path.dirname(new_full)), exist_ok=True)
        os.replace(_ext(old_full), _ext(new_full))

    def usage(self) -> int:
        """Bytes this workspace holds on disk now, whoever wrote them."""
        total = 0
        stack = [_ext(self.root)]
        while stack:
            try:
                with os.scandir(stack.pop()) as it:
                    entries = list(it)
            except OSError:
                continue
            for e in entries:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(st.st_mode) and not _is_link(st):
                    stack.append(e.path)
                else:
                    total += st.st_size
        return total

    def _walk_copy(self) -> tuple[dict[str, tuple[int, int]], list[str]]:
        """Regular files of the copy (path -> size, mtime_ns), and the links found, which are never entered."""
        found: dict[str, tuple[int, int]] = {}
        links: list[str] = []
        stack = [""]
        while stack:
            folder = stack.pop()
            try:
                with os.scandir(_ext(_join(self.copy, folder.rstrip("/")))) as it:
                    entries = list(it)
            except OSError:
                continue
            for e in entries:
                rel = folder + e.name
                if rel.lower() == RESERVED or e.name.lower() == ".git":       # a test may run `git init` in the copy; a .git there would be offered to the project
                    continue
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if _is_link(st):
                    links.append(rel)
                elif stat.S_ISDIR(st.st_mode):
                    stack.append(rel + "/")
                elif stat.S_ISREG(st.st_mode):
                    found[rel] = (st.st_size, st.st_mtime_ns)
        return found, links

    def _check_ignored(self, paths: list[str]) -> None:
        """Asks the project's own rules: the copy has no .git to ask. A path git cannot answer for stays visible, which errs on the side of showing it."""
        groups: dict[str, list[str]] = {}
        for rel in paths:
            self._ignored[rel] = False
            repo = max((r for r in self._repos if rel.startswith(r)), key=len, default="")
            groups.setdefault(repo, []).append(rel[len(repo):])
        for repo, names in groups.items():
            code, out, _ = _git(_join(self.project, repo.rstrip("/")), "check-ignore", "-z", "--stdin", stdin=os.fsencode("\0".join(names) + "\0"))
            if code in (0, 1):
                for name in out.split(b"\0"):
                    if name:
                        self._ignored[repo + os.fsdecode(name)] = True

    def _hidden(self, new: list[str]) -> set[str]:
        """Which of these files, unknown to the base, are what a build or a test left behind and not part of a change (design case 21)."""
        out = {p for p in new if _is_output(p)}
        ask = [p for p in new if p not in out and p not in self._ignored]
        if ask:
            self._check_ignored(ask)
        return out | {p for p in new if self._ignored.get(p)}

    def listing(self) -> dict[str, tuple[int, int]]:
        """What search may look at: the copy's files, without what the copy leaves out. Path -> size, mtime_ns."""
        self._check_open()
        found, _ = self._walk_copy()
        hidden = self._hidden([p for p in found if p not in self.base])
        return {p: v for p, v in found.items() if p not in hidden and not private(p)}

    # -------------------------------------------------------------------------------------------- the delta
    def _hash_copy(self, stamps: dict[str, tuple[int, int]], full: bool) -> dict[str, str | None]:
        """The hash of each of these files of the copy. A file with the size and mtime it had when last hashed is not opened, unless `full`."""
        out: dict[str, str | None] = {}
        todo = []
        for rel, stamp in stamps.items():
            known = self._hashes.get(rel)
            if known and known[0] == stamp and not full:
                out[rel] = known[1]
            else:
                todo.append(rel)
        now = time.time_ns()
        paths = [_join(self.copy, rel) for rel in todo]
        if len(paths) > WORKERS:
            with ThreadPoolExecutor(WORKERS) as pool:
                done = list(pool.map(_hash_file, paths))
        else:
            done = [_hash_file(p) for p in paths]
        for rel, digest in zip(todo, done):
            out[rel] = digest
            if digest is not None and stamps[rel][1] < now - MTIME_TICK_NS:
                self._hashes[rel] = (stamps[rel], digest)
        return out

    def changed(self, full: bool = False) -> list[dict]:
        """What differs from the base: {"path", "status", "hash", "base"} and "from" for a rename. M changed, A new, D deleted, R renamed with the same bytes, L a link in the copy (never offered).

        A file whose size and mtime have not moved counts as unchanged without being opened, which is what makes this cheap enough to run after every command. So code that restores both after an edit can hide it from
        this call; `full` reads every file, and `apply_plan` always does."""
        self._check_open()
        found, links = self._walk_copy()
        hidden = self._hidden([p for p in found if p not in self.base])
        hashes = self._hash_copy({p: v for p, v in found.items() if p not in hidden}, full)
        rows: dict[str, dict] = {}
        for rel in links:
            rows[rel] = {"path": rel, "status": "L", "hash": None, "base": None}
        for rel, now in hashes.items():
            kept = self.base.get(rel)
            if kept is None:
                rows[rel] = {"path": rel, "status": "A", "hash": now, "base": None}
            elif now != kept.hash:
                rows[rel] = {"path": rel, "status": "M", "hash": now, "base": kept.hash}
        gone: dict[str, list[str]] = {}
        for rel, kept in self.base.items():
            if rel not in found and rel not in rows:
                rows[rel] = {"path": rel, "status": "D", "hash": None, "base": kept.hash}
                if kept.size:                                                 # two empty files are no evidence of a rename
                    gone.setdefault(kept.hash, []).append(rel)
        for row in sorted(rows.values(), key=lambda r: r["path"]):
            if row["status"] == "A" and gone.get(row["hash"]):
                old = gone[row["hash"]].pop(0)
                row.update(status="R", base=row["hash"], **{"from": old})
                del rows[old]
        return sorted(rows.values(), key=lambda r: r["path"])

    @staticmethod
    def _delta_of(rows: list[dict]) -> str:
        h = hashlib.sha256()
        for r in rows:
            h.update(f"{r['path']}\0{r['status']}\0{r.get('from')}\0{r['hash']}\0{r['base']}\n".encode("utf-8", "surrogateescape"))
        return h.hexdigest()

    def _base_bytes(self, rel: str) -> bytes:
        kept = self.base[rel]
        with open(self._pack, "rb") as f:
            f.seek(kept.offset)
            return zlib.decompress(f.read(kept.length))

    def _diff_one(self, row: dict) -> tuple[str, int, int]:
        path, status = row["path"], row["status"]
        old = row.get("from", path)
        head = f"diff --git a/{old} b/{path}\n"
        if status == "L":
            return head + "a link in the copy: it is not offered to the project\n", 0, 0
        if status == "R":
            return head + f"similarity index 100%\nrename from {old}\nrename to {path}\n", 0, 0
        try:
            before = b"" if status == "A" else self._base_bytes(path)
            after = b"" if status == "D" else self.read_bytes(path, self.max_file + 1)
        except (OSError, WorkspaceError) as e:
            return head + f"it could not be read: {e}\n", 0, 0
        if len(after) > self.max_file:
            return head + "it is over the size limit: its text is not shown\n", 0, 0
        return self._diff_bytes(head, path, before, after, status)

    def _diff_bytes(self, head: str, path: str, before: bytes, after: bytes, status: str) -> tuple[str, int, int]:
        if status == "A":
            head += "new file mode 100644\n"
        elif status == "D":
            head += "deleted file mode 100644\n"
        src, dst = ("/dev/null" if status == "A" else f"a/{path}"), ("/dev/null" if status == "D" else f"b/{path}")
        if b"\0" in before[:8192] or b"\0" in after[:8192]:
            return head + f"Binary files {src} and {dst} differ\n", 0, 0
        left, right = _lines(before), _lines(after)
        if _changed_span(left, right) > MAX_DIFF_SPAN:                        # design case 20: a count, not the text
            had, has = Counter(left), Counter(right)
            added, removed = sum((has - had).values()), sum((had - has).values())
            return head + f"too much changed to show: {added:,} lines added, {removed:,} removed\n", added, removed
        out, added, removed = [head], 0, 0
        for i, line in enumerate(difflib.unified_diff(left, right, src, dst)):
            out.append(line)
            if i >= 2:                                                        # past the two file lines, which also start with - and +
                added += line.startswith("+")
                removed += line.startswith("-")
            if not line.endswith("\n"):
                out.append("\n\\ No newline at end of file\n")
        return "".join(out), added, removed

    def diff(self) -> str:
        """A unified diff of the delta. It is worked out here from the two sets of bytes, so no setting of git on this PC can change what the person approving it sees."""
        return "".join(self._diff_one(r)[0] for r in self.changed())

    def diff_stat(self) -> dict:
        """The size of the delta, for when it is too long to show (design case 20): {"files", "added", "removed"}."""
        rows = [self._diff_one(r) for r in self.changed()]
        return {"files": len(rows), "added": sum(r[1] for r in rows), "removed": sum(r[2] for r in rows)}

    # -------------------------------------------------------------------------------------------- freshness
    def _copy_hash(self, rel: str) -> str | None:
        """The hash of a file of the copy, None when it is not there. A link is its own answer: it can equal neither side, so it counts as the copy's change."""
        full = _join(self.copy, rel)
        st = _lstat(full)
        if st is None:
            return None
        if _is_link(st) or not _same(_real(full), full):
            return "link"
        return _hash_file(full)

    def _read_admitted(self, rel: str, tracked: bool) -> tuple[bytes | None, str | None]:
        """A file of the project if it may be in the copy; else (None, why it may not); (None, None) when it is not there."""
        st = _lstat(_join(self.project, rel))
        if st is None:
            return None, None
        why = self._left_out(rel, tracked, st)
        if why:
            return None, why
        data = self._read_real(rel)
        return (data, None) if data is not None else (None, "it could not be read")

    def _expand(self, rel: str, tracked: bool) -> dict[str, bool]:
        """A path of `git status` may be a whole submodule or nested repository: the files in it."""
        rel = rel.rstrip("/")
        if rel in self.base:
            return {rel: tracked}
        st = _lstat(_join(self.project, rel))
        if st is not None and not (stat.S_ISDIR(st.st_mode) and not _is_link(st)):
            return {rel: tracked}
        inside = {p: True for p in self.base if p.startswith(rel + "/")}
        if st is None:
            return inside or {rel: tracked}
        if _lstat(_join(self.project, rel + "/.git")) is not None:
            self._repos.add(rel + "/")
            inside.update({p: t for p, t, _ in self._walk_git(rel + "/")})
        return inside

    def _sync_one(self, rel: str, tracked: bool, report: dict, pack) -> None:
        data, why = self._read_admitted(rel, tracked)
        if why:
            self.excluded[rel] = why
        else:
            self.excluded.pop(rel, None)
        real = _hash(data) if data is not None else None
        kept = self.base.get(rel)
        base = kept.hash if kept else None
        if real == base:
            return
        here = self._copy_hash(rel)
        if here != real and here != base:
            report["conflicts"].append(rel)
            return
        if here == base:
            if data is not None:
                self._used = self.usage() if self._used is None else self._used
                if self._used + len(data) > self.quota:
                    report["skipped"].append(rel)
                    return
            try:
                if data is None:
                    os.remove(_ext(_join(self.copy, rel)))
                else:
                    self._put(rel, data, os.lstat(_ext(_join(self.project, rel))).st_mode)
                    self._used += len(data)
            except FileNotFoundError:
                pass
            except OSError:                                                   # the box holds it open or marked it read-only: the old base stays, so apply will refuse rather than guess
                report["skipped"].append(rel)
                return
            report["removed" if data is None else ("added" if base is None else "refreshed")].append(rel)
        if data is None:
            self.base.pop(rel, None)
        else:
            self.base[rel] = self._keep(pack, real, len(data), zlib.compress(data, 1))

    def sync(self) -> dict:
        """Brings the base up to the real project (design 4.4). A file the copy has not touched follows the project in silence. A file changed on both sides is a conflict: neither side is overwritten and the base stays, so apply refuses it until it is settled.

        `git status` only names candidates, and each is then compared by hash: a file that was listed last time and is clean now has moved as well. So does everything, potentially, when HEAD moved.
        {"refreshed", "added", "removed", "conflicts", "skipped": paths; "head": {"from", "to"} when HEAD moved, else None}. `skipped`: it could not follow (the quota, or the box holds it); the old base stays, so apply refuses it."""
        self._check_open()
        self._parents, self._ignored = {}, {}
        self._used = None
        head, status = self._status()
        report: dict = {"refreshed": [], "added": [], "removed": [], "conflicts": [], "skipped": [], "head": None}
        names: dict[str, bool] = {p: True for p in self._dirty}
        names.update(status)
        if head != self.head:
            report["head"] = {"from": self.head, "to": head}
            names.update({p: True for p in self.base})
            names.update({p: t for p, t, _ in self._walk_git()})
        todo: dict[str, bool] = {}
        for rel, tracked in names.items():
            for one, known in self._expand(rel, tracked).items():
                todo[one] = todo.get(one, False) or known
        with open(self._pack, "ab+") as pack:
            for rel in sorted(todo):
                self._sync_one(rel, todo[rel] or rel in self.base, report, pack)
        self._dirty = set(status) | set(report["conflicts"]) | set(report["skipped"])
        self.conflicts = set(report["conflicts"])
        self.head = head
        return report

    # -------------------------------------------------------------------------------------------- settling a conflict
    @staticmethod
    def _seen(digest: str | None, hidden: str | None) -> str:
        """What names the project's side of a conflict to the model. A file it may not see gets no hash either: part of the hash of a short secret is part of the secret."""
        return "hidden" if hidden else digest[:16] if digest else "none"

    def _both_changed(self, rel: str) -> tuple[bool, str | None, str | None, str | None]:
        """(whether both sides moved from the base and from each other, the hash of the project's file, of the base, of the copy)."""
        _, blocked, real = self._project_state(rel)
        if blocked:
            real = "blocked"                                                  # a link where a file was: it cannot equal either side
        kept = self.base.get(rel)
        base = kept.hash if kept else None
        here = self._copy_hash(rel)
        return real != base and here != base and here != real, real, base, here

    def conflict(self, rel: str) -> dict | None:
        """The two sides of a conflicted file against the base (design 4.4); None when it is in no conflict now. It reads the real project, which decision Q9 allows, and writes nothing.
        What the copy leaves out stays out here: a secret, a link or a file past the size limit is named and its bytes are not shown. {"path", "seen": what `resolve` has to be given back, "real": diff, "copy": diff, "hidden": why the real side is not shown}."""
        self._check_open()
        why = bad_path(rel)
        if why:
            raise WorkspaceError(f"{rel}: {why}")
        self._parents = {}
        both, real, base, here = self._both_changed(rel)
        if not both:
            return None
        data, hidden = self._read_admitted(rel, True)
        kept = self._base_bytes(rel) if rel in self.base else b""
        head = f"diff --git a/{rel} b/{rel}\n"
        real_diff = ""
        if not hidden:
            real_diff = self._diff_bytes(head, rel, kept, data or b"", "D" if data is None else ("A" if base is None else "M"))[0]
        copy_bytes = b"" if here in (None, "link") else self.read_bytes(rel, self.max_file + 1)
        copy_diff = "a link in the copy\n" if here == "link" else self._diff_bytes(head, rel, kept, copy_bytes, "D" if here is None else ("A" if base is None else "M"))[0]
        return {"path": rel, "seen": self._seen(real, hidden), "real": real_diff, "copy": copy_diff, "hidden": hidden}

    def _settle(self, rel: str, data: bytes | None) -> None:
        """The base of this file is now what the project holds."""
        if data is None:
            self.base.pop(rel, None)
        else:
            with open(self._pack, "ab+") as pack:
                self.base[rel] = self._keep(pack, _hash(data), len(data), zlib.compress(data, 1))
        self.conflicts.discard(rel)

    def _merge(self, mine: bytes, base: bytes, theirs: bytes) -> tuple[bytes, bool] | None:
        """(the three-way merge, whether both sides changed the same lines, in which case `mine` wins them); None for what git will not merge, such as a binary file. The files sit outside the copy, so the box cannot change what is merged."""
        folder = tempfile.mkdtemp(prefix="merge-", dir=self.root)
        try:
            for name, data in (("mine", mine), ("base", base), ("theirs", theirs)):
                with open(os.path.join(folder, name), "wb") as f:
                    f.write(data)
            code, out, _ = _git(folder, "merge-file", "-p", "mine", "base", "theirs")
            if code == 0:
                return out, False
            if not 0 < code < 128:
                return None
            code, out, _ = _git(folder, "merge-file", "-p", "--ours", "mine", "base", "theirs")
            return (out, True) if code == 0 else None
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def resolve(self, rel: str, take: str, seen: str) -> dict:
        """Settles one conflicted file so that apply accepts it, with nothing written to the real project. `seen`: what `conflict` showed; a project file that is not that one any more raises ConflictMoved, and nothing changes.

        "real" puts the project's file in the copy and drops this file's part of the change. "copy" keeps the copy's change and moves its base to the project's file: the change is replayed on it by a three-way merge, so that what the real side did elsewhere in the file stays, and where both sides changed the same lines the copy's lines win.
        The delta is not the one that was tested after this, so a test record does not carry over. {"path", "take", "merged": the file now holds changes of both sides, "overlap": both sides changed the same lines}."""
        self._check_open()
        if take not in ("real", "copy"):
            raise WorkspaceError(f"{rel}: take is real or copy")
        why = bad_path(rel)
        if why:
            raise WorkspaceError(f"{rel}: {why}")
        self._parents = {}
        both, real, base, here = self._both_changed(rel)
        data, hidden = self._read_admitted(rel, True)
        if self._seen(real, hidden) != seen or (data is not None and _hash(data) != real):
            raise ConflictMoved(f"{rel}: the real file changed again after it was shown, so nothing was settled")
        if not both:
            raise WorkspaceError(f"{rel}: it is in no conflict now")
        full = self._in_real_folder(rel)
        out = {"path": rel, "take": take, "merged": False, "overlap": False}
        if take == "real":
            if hidden:
                self.excluded[rel] = hidden
            if here == "link" or (data is None and _lstat(full) is not None):
                os.unlink(_ext(full))
            if data is not None:
                if self.usage() + len(data) > self.quota:
                    raise QuotaExceeded(f"{rel}: taking it would pass the quota of {self.quota:,}")
                self._put(rel, data, os.lstat(_ext(_join(self.project, rel))).st_mode)
        else:
            if hidden:
                raise WorkspaceError(f"{rel}: {hidden}, so the copy cannot be set on top of it; only take=real settles it")
            if here == "link":
                raise WorkspaceError(f"{rel}: the copy holds a link there, which cannot go to the project")
            if rel in self.base and data is not None and here is not None:
                mine = self.read_bytes(rel, self.max_file + 1)
                done = self._merge(mine, self._base_bytes(rel), data)
                if done:
                    out.update(merged=done[0] != mine, overlap=done[1])
                    if done[0] != mine:
                        self.write_bytes(rel, done[0])
        self._settle(rel, data)
        return out

    # -------------------------------------------------------------------------------------------- back to the project
    def base_id(self) -> str:
        """Names the base as a whole: two runs with the same id ran over the same project."""
        h = hashlib.sha256()
        for rel in sorted(self.base):
            h.update(f"{rel}\0{self.base[rel].hash}\n".encode("utf-8", "surrogateescape"))
        return h.hexdigest()

    def export_base(self, target: str) -> None:
        """Writes the base to an empty folder, from the kept bytes: the project may have moved on, and it is the base that a change is measured against."""
        self._check_open()

        def one(rel: str) -> None:
            with open(_ext(_join(target, rel)), "wb") as f:
                f.write(self._base_bytes(rel))

        for folder in {os.path.dirname(_join(target, rel)) for rel in self.base} | {target}:
            os.makedirs(_ext(folder), exist_ok=True)
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(one, self.base))

    def base_bytes(self, rel: str) -> bytes | None:
        """What a file was at the base, None when it was not there: the way back from a failed apply."""
        return self._base_bytes(rel) if rel in self.base else None

    def tested_record(self, command: str, exit_code: int, summary: str, simulated: str | None = None) -> dict:
        """Notes a test run against the delta as it is now (design 4.7). `simulated`: what the box replaced, such as "no GPU, fake model" (design 12): that label has to reach the approval, since a simulated pass proves the logic only.
        `project`: the base it ran over. A file the delta does not touch can follow the real project afterwards, and then what was tested is not what would be applied, though the delta is the same."""
        rows = self.changed()
        record = {"delta": self._delta_of(rows), "files": [r["path"] for r in rows], "base": {r["path"]: r["base"] for r in rows}, "command": command, "exit_code": exit_code,
                  "summary": summary, "simulated": simulated, "project": self.base_id(), "time": time.time()}
        self.tested.append(record)
        del self.tested[:-KEEP_RECORDS]
        return record

    def _project_state(self, rel: str) -> tuple[str, str | None, str | None]:
        """(the path in the project, why it cannot be written or None, the hash of the file there or None when there is none)."""
        full = _join(self.project, rel)
        if not _same(_real(full), full):                                      # a link in the project, or a short name such as GIT~1 that leads to .git
            return full, "it leads through a link in the project", None
        st = _lstat(full)
        if st is None:
            return full, None, None
        if not stat.S_ISREG(st.st_mode):
            return full, "it is not a regular file in the project", None
        return full, None, _hash_file(full)

    def _refusal(self, rel: str, expected: str | None, reasons: list[dict]) -> str:
        full, why, now = self._project_state(rel)
        if not why and now != expected:
            why = "it changed in the real project" if expected else "a file already exists there in the real project"
        if why:
            reasons.append({"path": rel, "why": why})
        return full

    def apply_plan(self) -> Plan:
        """The steps that would put the delta in the real project, all of them or none: raises ApplyRefused when any file cannot go. It writes nothing, not even to the copy.

        The bytes are in the plan, read once and checked against the hashes the delta is made of, so a process still running in the box cannot change what was approved. The file's hash in the project is checked
        here and has to be checked again with `recheck` after the user has answered, because the question can take minutes."""
        rows = self.changed(full=True)
        reasons: list[dict] = []
        writes: list[Write] = []
        deletes: list[Write] = []
        fold = {os.path.normcase(p): p for p in self.base}
        kept_names = {os.path.normcase(r["path"]) for r in rows if r["status"] != "D"}
        for row in rows:
            rel, status = row["path"], row["status"]
            why = "it is a link in the copy" if status == "L" else bad_path(rel)
            if why:
                reasons.append({"path": rel, "why": why})
                continue
            if status != "D":
                owner = self.base.get(rel) or self.base.get(fold.get(os.path.normcase(rel), ""))         # a name changed only in case is the same file on Windows: it has to be compared with that file
                expected = owner.hash if owner else None
                full = self._refusal(rel, expected, reasons)
                try:
                    data = self.read_bytes(rel, self.max_file + 1)
                except (OSError, WorkspaceError) as e:
                    reasons.append({"path": rel, "why": f"it could not be read: {e}"})
                    continue
                if len(data) > self.max_file:
                    reasons.append({"path": rel, "why": f"it is over the size limit ({len(data):,} bytes)"})
                elif _hash(data) != row["hash"]:
                    reasons.append({"path": rel, "why": "it is still changing in the copy"})
                else:
                    writes.append(Write(rel, full, "write", data, expected))
            gone = rel if status == "D" else row.get("from")
            if gone and os.path.normcase(gone) not in kept_names:             # deleting the old spelling of a name changed only in case would delete the new one
                deletes.append(Write(gone, self._refusal(gone, self.base[gone].hash, reasons), "delete", None, self.base[gone].hash))
        if reasons:
            raise ApplyRefused(reasons)
        delta = self._delta_of(rows)
        tested = next((r for r in reversed(self.tested) if r["delta"] == delta), None)
        return Plan(writes + deletes, delta, tested)

    def recheck(self, plan: Plan) -> None:
        """Raises ApplyRefused when the project has moved since the plan was made. For the moment after the user answers and before the first write."""
        reasons: list[dict] = []
        for w in plan.writes:
            self._refusal(w.path, w.base, reasons)
        if reasons:
            raise ApplyRefused(reasons)

    def applied(self, plan: Plan, steps: list[Write] | None = None) -> None:
        """The base follows what was written. Without this a `sync` right after would call a file a conflict when a formatter or the user changed it the moment it was written.
        `steps`: the ones that took place, when not all of them did."""
        for w in plan.writes if steps is None else steps:
            self._settle(w.path, w.data if w.action == "write" else None)

    # -------------------------------------------------------------------------------------------- the end
    def close(self) -> bool:
        """Removes the scratch copy. Whether it is gone: an antivirus scan or a process still in the box can hold a file for a moment, so it tries a few times, and never raises."""
        self.closed = True
        for wait in (0.05, 0.2, 0.5, 0):
            shutil.rmtree(_ext(self.root), onerror=_force)
            if not os.path.lexists(_ext(self.root)):
                return True
            time.sleep(wait)
        return False
