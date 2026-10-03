import type { L } from "@/lib/i18n";
import type { SceneId } from "@/components/scenes";
import type { ChartId } from "./numbers";
import { file, issue, pr } from "./sources";

export type Status = "shipped" | "off-by-default" | "opt-in";
export type Category = "agent" | "setup" | "see" | "api" | "engine";

/** A piece of the app's interface, played as a scene (components/scenes), with the sentence a screen reader gets for it. */
export type FeatureScene = {
  scene: SceneId;
  alt: L;
  label?: L;
  /** Show the app in this language whatever language the page is in. */
  lang?: "en" | "th";
};
export type Evidence = { label: string; href: string };

export type Feature = {
  id: string;
  category: Category;
  title: L;
  summary: L;
  points: L[];
  status: Status;
  /** One scene of the app, or a few that can be switched. None = a typographic panel. */
  scenes?: FeatureScene[];
  evidence: Evidence[];
  /** A comparison chart of a measured result that this feature claims. */
  chart?: ChartId;
  /** Conditions that belong to a claim made in this feature. */
  note?: L;
};

export const CATEGORIES: { id: Category; label: L }[] = [
  { id: "agent", label: { en: "Coding tools", th: "เครื่องมือเขียนโค้ด" } },
  { id: "setup", label: { en: "Your setup", th: "สิ่งที่คุณมีอยู่แล้ว" } },
  { id: "see", label: { en: "See what happens", th: "เห็นสิ่งที่เกิดขึ้น" } },
  { id: "api", label: { en: "API", th: "API" } },
  { id: "engine", label: { en: "Engine", th: "เอนจิน" } },
];

export const STATUS_LABEL: Record<Status, L> = {
  shipped: { en: "Shipped", th: "ใช้งานได้แล้ว" },
  "off-by-default": { en: "Off by default", th: "ปิดไว้เป็นค่าเริ่มต้น" },
  "opt-in": { en: "Opt-in", th: "เปิดเองเมื่อต้องการ" },
};

const Sc = (scene: SceneId, alt: L, label?: L, lang?: "en" | "th"): FeatureScene => ({ scene, alt, label, lang });

export const FEATURES: Feature[] = [
  {
    id: "chat-tools",
    category: "agent",
    status: "shipped",
    title: { en: "Coding tools in the chat", th: "เครื่องมือเขียนโค้ดในแชต" },
    summary: {
      en: "The chat reads, searches and changes files and runs commands in your project, with the tool names and parameters Claude Code uses. The model here is a local one.",
      th: "แชตอ่าน ค้นหา แก้ไฟล์ และรันคำสั่งในโปรเจกต์ของคุณได้ ด้วยชื่อเครื่องมือและพารามิเตอร์เดียวกับ Claude Code และโมเดลที่ใช้เป็นโมเดลบนเครื่องคุณเอง",
    },
    points: [
      {
        en: "15 tools: Read, Write, Edit, Glob, Grep, TodoWrite, ExitPlanMode, NotebookEdit, AskUserQuestion, Bash, BashOutput, KillShell, WebFetch, WebSearch, Task.",
        th: "15 เครื่องมือ: Read, Write, Edit, Glob, Grep, TodoWrite, ExitPlanMode, NotebookEdit, AskUserQuestion, Bash, BashOutput, KillShell, WebFetch, WebSearch, Task",
      },
      {
        en: "Commands run in Git Bash or PowerShell in the project folder, with a timeout, background mode, a `cd` that carries over, and a kill for the whole process tree.",
        th: "คำสั่งรันใน Git Bash หรือ PowerShell ที่โฟลเดอร์ของโปรเจกต์ มี timeout โหมด background และสั่งหยุดได้ทั้งต้นไม้ของโปรเซส",
      },
      {
        en: "Plans written with TodoWrite appear as a checklist in the conversation and in the side panel.",
        th: "แผนที่โมเดลเขียนด้วย TodoWrite แสดงเป็นเช็กลิสต์ทั้งในบทสนทนาและแผงด้านข้าง",
      },
    ],
    scenes: [
      Sc("chat-run", {
        en: "The chat after a real run: Read, Edit and Bash cards, then the answer with the measured speed underneath.",
        th: "แชตหลังรันจริง: การ์ด Read, Edit และ Bash แล้วตามด้วยคำตอบ พร้อมความเร็วที่วัดได้ด้านล่าง",
      }),
    ],
    note: {
      en: "Web access (WebFetch, WebSearch) and sub-agents (Task) are off until you switch them on.",
      th: "การเข้าเว็บ (WebFetch, WebSearch) และ sub-agent (Task) ปิดอยู่จนกว่าคุณจะเปิดเอง",
    },
    evidence: [
      { label: "serve/agent.py", href: file("serve/agent.py") },
      { label: "serve/shell.py", href: file("serve/shell.py") },
      { label: "test_agent.py · 71 tests", href: file("serve/test_agent.py") },
      { label: "test_shell.py · 29 tests", href: file("serve/test_shell.py") },
      { label: "PR #104", href: pr(104) },
    ],
  },
  {
    id: "ask-first",
    category: "agent",
    status: "shipped",
    title: { en: "It asks before it acts", th: "ถามก่อนลงมือ" },
    summary: {
      en: "Files inside the project folder are free to read and change. Everything else asks you first, in a card that says what would happen.",
      th: "ไฟล์ในโฟลเดอร์ของโปรเจกต์อ่านและแก้ได้เลย ส่วนอย่างอื่นจะถามก่อนด้วยการ์ดที่บอกว่าจะเกิดอะไรขึ้น",
    },
    points: [
      {
        en: "Three modes: Ask, Plan and Auto. Auto adds a second opinion: a side request rates each call from 1 to 5 for safety and never sees tool output.",
        th: "สามโหมด: Ask, Plan และ Auto โดย Auto เพิ่มความเห็นที่สอง คือคำขอแยกที่ให้คะแนนความปลอดภัยของแต่ละคำสั่ง 1–5 และไม่เคยเห็นผลลัพธ์ของเครื่องมือ",
      },
      {
        en: "Secrets, changes inside .git and commands that are hard to undo ask every time. No rule, hook or judge can allow them.",
        th: "ความลับ การแก้ใน .git และคำสั่งที่ย้อนกลับยากจะถามทุกครั้ง ไม่มีกฎ hook หรือตัวตัดสินใดอนุญาตแทนได้",
      },
      {
        en: "The card names the command and why it asks, and keeps your answer for this chat, this project or everywhere.",
        th: "การ์ดบอกคำสั่งและเหตุผลที่ถาม และจำคำตอบของคุณไว้ได้ทั้งสำหรับแชตนี้ โปรเจกต์นี้ หรือทุกที่",
      },
    ],
    scenes: [
      Sc("permission",
        {
          en: "A permission card: “Run this command?” with the command, the reason it asks, and Allow / Deny.",
          th: "การ์ดขออนุญาตจริง: “Run this command?” พร้อมคำสั่ง เหตุผลที่ถาม และปุ่ม Allow / Deny",
        },
        { en: "Permission card", th: "การ์ดขออนุญาต" },
      ),
      Sc("modes",
        {
          en: "The coding-mode chip on the prompt bar opens Ask, Plan and Auto.",
          th: "ชิปโหมดเขียนโค้ดบนแถบพิมพ์ เปิดตัวเลือก Ask, Plan และ Auto",
        },
        { en: "Ask · Plan · Auto", th: "Ask · Plan · Auto" },
      ),
    ],
    evidence: [
      { label: "serve/permissions.py · 44 tests", href: file("serve/permissions.py") },
      { label: "serve/judge.py", href: file("serve/judge.py") },
      { label: "serve/agent_run.py", href: file("serve/agent_run.py") },
      { label: "test_agent_run.py · 21 tests", href: file("serve/test_agent_run.py") },
    ],
  },
  {
    id: "rules",
    category: "agent",
    status: "shipped",
    title: { en: "Permissions you can write down", th: "สิทธิ์ที่เขียนเป็นกฎได้" },
    summary: {
      en: "Keep an answer for good as a rule, in Claude Code's own syntax, per project or everywhere.",
      th: "เก็บคำตอบไว้ถาวรเป็นกฎ ด้วยไวยากรณ์เดียวกับ Claude Code แยกตามโปรเจกต์หรือใช้ทุกที่",
    },
    points: [
      {
        en: "Bash(npm test:*) for commands that start with npm test, Read(src/**) for files under src, Edit(docs/**), or a bare tool name.",
        th: "Bash(npm test:*) สำหรับคำสั่งที่ขึ้นต้นด้วย npm test, Read(src/**) สำหรับไฟล์ใต้ src, Edit(docs/**) หรือชื่อเครื่องมือเฉยๆ",
      },
      { en: "“Never” wins over “Allow”.", th: "“Never” ชนะ “Allow” เสมอ" },
      {
        en: "Hooks (before a call, after a call, on prompt, on stop) are written by hand in the run config, so a web page can never make Strata run a command.",
        th: "Hook (ก่อนเรียก หลังเรียก เมื่อส่งข้อความ เมื่อจบ) เขียนด้วยมือใน run config จึงไม่มีทางที่หน้าเว็บจะสั่งให้ Strata รันคำสั่งได้",
      },
    ],
    scenes: [
      Sc("permissions", {
        en: "Settings › Permissions: rules for everywhere and for this project, with Allow and Never.",
        th: "Settings › Permissions: กฎสำหรับทุกที่และสำหรับโปรเจกต์นี้ ทั้ง Allow และ Never",
      }),
    ],
    evidence: [
      { label: "serve/permissions.py", href: file("serve/permissions.py") },
      { label: "serve/hooks.py · 46 tests", href: file("serve/hooks.py") },
      { label: "test_permissions.py", href: file("serve/test_permissions.py") },
    ],
  },
  {
    id: "projects",
    category: "agent",
    status: "shipped",
    title: { en: "Projects with more than one folder", th: "โปรเจกต์ที่มีได้มากกว่าหนึ่งโฟลเดอร์" },
    summary: {
      en: "A project is one or more folders on your PC, such as the worktrees of one repository. The first is the main folder: commands run there.",
      th: "โปรเจกต์คือหนึ่งโฟลเดอร์หรือมากกว่าบนเครื่องคุณ เช่น worktree ของ repo เดียวกัน โฟลเดอร์แรกเป็นโฟลเดอร์หลักที่คำสั่งจะรัน",
    },
    points: [
      {
        en: "Add as many folders as the work needs, mark a different one as main, remove one — and the tools work in all of them.",
        th: "เพิ่มกี่โฟลเดอร์ก็ได้ตามงาน เปลี่ยนโฟลเดอร์หลัก หรือเอาออก และเครื่องมือทำงานได้ในทุกโฟลเดอร์",
      },
      {
        en: "Type a path and the folders that match are offered; or browse.",
        th: "พิมพ์พาธแล้วระบบเสนอโฟลเดอร์ที่ตรงกัน หรือกดเลือกเอง",
      },
      {
        en: "Chats can be started inside a project and are listed under it; each project keeps its own permission rules.",
        th: "เริ่มแชตในโปรเจกต์ได้และแสดงอยู่ใต้โปรเจกต์นั้น แต่ละโปรเจกต์มีกฎสิทธิ์ของตัวเอง",
      },
    ],
    scenes: [
      Sc("project", {
        en: "The New project dialog with two folders: a repository and its git worktree, the first marked Main.",
        th: "หน้าต่างสร้างโปรเจกต์ที่มีสองโฟลเดอร์: repo และ git worktree ของมัน โดยอันแรกเป็น Main",
      }),
    ],
    evidence: [
      { label: "serve/folders.py · 10 tests", href: file("serve/folders.py") },
      { label: "NewProjectDialog.tsx", href: file("serve/ui/src/components/NewProjectDialog.tsx") },
      { label: "FoldersEditor.tsx", href: file("serve/ui/src/components/FoldersEditor.tsx") },
      { label: "lib/folders.test.ts", href: file("serve/ui/src/lib/folders.test.ts") },
    ],
  },
  {
    id: "panel",
    category: "agent",
    status: "shipped",
    title: { en: "A side panel that knows your repo", th: "แผงด้านข้างที่รู้จัก repo ของคุณ" },
    summary: {
      en: "Git, Plan, Skills, Memory and Context beside the conversation, for the project the chat works in.",
      th: "Git, Plan, Skills, Memory และ Context อยู่ข้างบทสนทนา สำหรับโปรเจกต์ที่แชตทำงานอยู่",
    },
    points: [
      {
        en: "Git: branch, changes, branches, worktrees, recent commits and file diffs. Read-only, with a hardened git call.",
        th: "Git: branch ไฟล์ที่เปลี่ยน branches, worktrees, commit ล่าสุด และ diff ของไฟล์ อ่านอย่างเดียวและเรียก git แบบป้องกันไว้",
      },
      { en: "Plan: the steps the model wrote with TodoWrite.", th: "Plan: ขั้นตอนที่โมเดลเขียนด้วย TodoWrite" },
      {
        en: "Memory: the project's own CLAUDE.md is always on; other apps' instruction files are listed and off until you switch them on.",
        th: "Memory: CLAUDE.md ของโปรเจกต์เปิดเสมอ ส่วนไฟล์คำสั่งของแอปอื่นแสดงรายการไว้และปิดอยู่จนกว่าคุณจะเปิด",
      },
    ],
    scenes: [
      Sc("git", { en: "Git tab: branch, not-staged changes, branches, worktrees, recent commits.", th: "แท็บ Git: branch ไฟล์ที่ยังไม่ stage, branches, worktrees, commit ล่าสุด" }, { en: "Git", th: "Git" }),
      Sc("plan", { en: "Plan tab: the three steps the model planned.", th: "แท็บ Plan: สามขั้นตอนที่โมเดลวางแผน" }, { en: "Plan", th: "Plan" }),
      Sc("memory", { en: "Memory tab: the project's CLAUDE.md is on; other apps' files are off.", th: "แท็บ Memory: CLAUDE.md ของโปรเจกต์เปิดอยู่ ไฟล์ของแอปอื่นปิดอยู่" }, { en: "Memory", th: "Memory" }),
      Sc("context", { en: "Context tab: tokens used, by part, and the automatic compaction point.", th: "แท็บ Context: โทเคนที่ใช้แยกตามส่วน และจุดที่ย่อบทสนทนาอัตโนมัติ" }, { en: "Context", th: "Context" }),
    ],
    evidence: [
      { label: "serve/gitview.py · 23 tests", href: file("serve/gitview.py") },
      { label: "serve/memory.py · 22 tests", href: file("serve/memory.py") },
      { label: "SidePanel.tsx", href: file("serve/ui/src/components/SidePanel.tsx") },
      { label: "ContextPanel.tsx", href: file("serve/ui/src/components/ContextPanel.tsx") },
    ],
  },
  {
    id: "import",
    category: "setup",
    status: "shipped",
    title: { en: "Bring your setup in a click", th: "นำสิ่งที่คุณมีอยู่แล้วมาใช้ในคลิกเดียว" },
    summary: {
      en: "Strata finds the skills, MCP servers and memory files that your other coding apps already use and lets you use them here.",
      th: "Strata ค้นหา skill, MCP server และไฟล์ memory ที่แอปเขียนโค้ดอื่นของคุณใช้อยู่แล้ว และให้นำมาใช้ที่นี่ได้",
    },
    points: [
      {
        en: "Reads Claude Code, Codex, Antigravity, Gemini CLI, Cursor, Claude Desktop and the shared ~/.agents folder. A read-only scan; Rescan any time.",
        th: "อ่านจาก Claude Code, Codex, Antigravity, Gemini CLI, Cursor, Claude Desktop และโฟลเดอร์ ~/.agents ที่ใช้ร่วมกัน สแกนแบบอ่านอย่างเดียว และสแกนใหม่ได้ทุกเมื่อ",
      },
      {
        en: "Skills: found in each app and in use until you switch that app off. The chat loads one when a task fits (`/` lists them).",
        th: "Skills: พบในแต่ละแอปและใช้ได้ทันทีจนกว่าคุณจะปิดแอปนั้น แชตโหลดมาใช้เมื่องานเข้ากัน (พิมพ์ `/` เพื่อดูรายการ)",
      },
      {
        en: "MCP servers: nothing runs until you press Import on one. It is copied into Strata's own list, where you can edit, turn off or delete it. Or paste from Claude Desktop.",
        th: "MCP server: ไม่มีอะไรรันจนกว่าคุณกด Import ที่ตัวนั้น มันถูกคัดลอกเข้ารายการของ Strata ซึ่งแก้ ปิด หรือลบได้ หรือวางจาก Claude Desktop",
      },
      {
        en: "Memory: your CLAUDE.md, AGENTS.md and per-project memory are read as those apps read them — each off until you switch it on, and they never change what the chat may do.",
        th: "Memory: CLAUDE.md, AGENTS.md และ memory ของแต่ละโปรเจกต์ถูกอ่านเหมือนที่แอปเหล่านั้นอ่าน — แต่ละอันปิดอยู่จนกว่าคุณจะเปิด และไม่เปลี่ยนสิ่งที่แชตทำได้",
      },
    ],
    note: {
      en: "This page shows no screenshot of the Import pages on purpose: they list what is on the developer's own PC.",
      th: "หน้านี้ไม่แสดงภาพของหน้า Import โดยตั้งใจ เพราะหน้านั้นแสดงรายการที่อยู่ในเครื่องของนักพัฒนาเอง",
    },
    evidence: [
      { label: "serve/harness.py · 78 tests", href: file("serve/harness.py") },
      { label: "serve/mcp_admin.py · 16 tests", href: file("serve/mcp_admin.py") },
      { label: "serve/skills.py", href: file("serve/skills.py") },
      { label: "serve/memory.py", href: file("serve/memory.py") },
    ],
  },
  {
    id: "status",
    category: "see",
    status: "shipped",
    title: { en: "A status that never goes quiet", th: "สถานะที่ไม่เคยเงียบ" },
    summary: {
      en: "At every moment of an answer there is one status in words for what is going on now, and a change is a change of words, never a gap.",
      th: "ทุกช่วงของการตอบมีสถานะเป็นคำหนึ่งบรรทัดว่ากำลังเกิดอะไรขึ้น และการเปลี่ยนขั้นคือการเปลี่ยนคำ ไม่ใช่ช่วงเงียบ",
    },
    points: [
      {
        en: "Writing the call to Bash, running Bash, running your hook, waiting for you, reading the tool's result, planning the next step, thinking, answering, a helper is working.",
        th: "กำลังเขียนคำสั่ง Bash, รัน Bash, รัน hook ของคุณ, รอคุณ, อ่านผลของเครื่องมือ, วางแผนขั้นต่อไป, คิด, ตอบ, มีตัวช่วยกำลังทำงาน",
      },
      {
        en: "The token count runs live, the thinking of each round sits under the tools of the round before, and a refreshed page re-attaches to the running answer.",
        th: "จำนวนโทเคนวิ่งสด ความคิดของแต่ละรอบอยู่ใต้เครื่องมือของรอบก่อน และรีเฟรชหน้าแล้วกลับมาต่อกับคำตอบที่กำลังรันอยู่ได้",
      },
      {
        en: "A step with no status, or one that only shows after it ends, breaks the project's own design rule.",
        th: "ขั้นตอนที่ไม่มีสถานะ หรือสถานะที่โผล่หลังจบขั้นเท่านั้น ถือว่าผิดกฎการออกแบบของโปรเจกต์เอง",
      },
    ],
    scenes: [
      Sc("status", {
        en: "Mid-run: Read and Edit are done, and the status line says “Reading the tool's result… 363 tokens”.",
        th: "ระหว่างรัน: Read และ Edit เสร็จแล้ว บรรทัดสถานะบอกว่า “Reading the tool's result… 363 tokens”",
      }),
    ],
    evidence: [
      { label: "lib/status.ts", href: file("serve/ui/src/lib/status.ts") },
      { label: "lib/status.test.ts", href: file("serve/ui/src/lib/status.test.ts") },
      { label: "serve/runs.py", href: file("serve/runs.py") },
      { label: "test_runs.py · 21 tests", href: file("serve/test_runs.py") },
      { label: "AGENTS.md · design principle", href: file("AGENTS.md") },
    ],
  },
  {
    id: "marks",
    category: "see",
    status: "shipped",
    title: { en: "Marks you can change", th: "เครื่องหมายสถานะที่เปลี่ยนได้" },
    summary: {
      en: "The mark that shows the model is working is yours to pick: orbs, loaders of dots, or an avatar with a mood of its own.",
      th: "เครื่องหมายที่บอกว่าโมเดลกำลังทำงานเป็นของคุณที่จะเลือก: orb, ตัวโหลดจุด หรือ avatar ที่มีอารมณ์ของตัวเอง",
    },
    points: [
      {
        en: "Four styles: Orbs, Orbs + Loading, Loading only, Avatar. The picker is in the top bar and in Settings.",
        th: "สี่แบบ: Orbs, Orbs + Loading, Loading only, Avatar เลือกได้จากแถบบนและหน้า Settings",
      },
      {
        en: "18 avatar shapes: one for every status, a shape for each status, or a random one for each place.",
        th: "avatar 18 รูปทรง: ใช้แบบเดียวทุกสถานะ แยกรูปทรงตามสถานะ หรือสุ่มต่อหนึ่งตำแหน่ง",
      },
      {
        en: "An avatar hops while the model works, looks around when the server is idle, and sleeps when the model is unloaded.",
        th: "avatar กระโดดเมื่อโมเดลทำงาน เหลียวมองเมื่อเซิร์ฟเวอร์ว่าง และหลับเมื่อโมเดลถูกถอนออก",
      },
    ],
    scenes: [
      Sc("marks", { en: "The Marks picker: Avatar is chosen, then the shape is changed one after another and the avatar changes with it.", th: "ตัวเลือก Marks: เลือก Avatar แล้วเปลี่ยนรูปทรงไปทีละแบบ และ avatar เปลี่ยนตาม" }),
    ],
    note: {
      en: "The orbs and avatars come from MIT-licensed libraries by Libraries.dev, vendored in the app (see Credits).",
      th: "orb และ avatar มาจากไลบรารีสัญญาอนุญาต MIT ของ Libraries.dev ที่ฝังไว้ในแอป (ดูส่วนเครดิต)",
    },
    evidence: [
      { label: "lib/orbs.ts", href: file("serve/ui/src/lib/orbs.ts") },
      { label: "lib/avatar.ts", href: file("serve/ui/src/lib/avatar.ts") },
      { label: "lib/avatar.test.ts", href: file("serve/ui/src/lib/avatar.test.ts") },
      { label: "StatusMarks.tsx", href: file("serve/ui/src/components/StatusMarks.tsx") },
    ],
  },
  {
    id: "controls",
    category: "agent",
    status: "shipped",
    title: { en: "Chat controls that keep up with a developer", th: "ตัวควบคุมแชตที่ตามนักพัฒนาทัน" },
    summary: {
      en: "The small things a coding session needs, so the conversation does not have to stop.",
      th: "ของเล็กๆ ที่การเขียนโค้ดต้องใช้ เพื่อไม่ให้บทสนทนาต้องหยุด",
    },
    points: [
      {
        en: "Rewind: every prompt is a checkpoint. Files the tools changed are put back; a file someone else changed is left alone.",
        th: "Rewind: ทุกข้อความคือ checkpoint ไฟล์ที่เครื่องมือแก้จะถูกคืนค่า ส่วนไฟล์ที่คนอื่นแก้จะไม่ถูกแตะ",
      },
      {
        en: "`@file` mentions complete from the project (secrets skipped). A message sent while the agent works is read at its next step.",
        th: "พิมพ์ `@file` เพื่อเติมชื่อไฟล์จากโปรเจกต์ (ข้ามไฟล์ความลับ) ข้อความที่ส่งระหว่างที่ agent ทำงานจะถูกอ่านในขั้นถัดไป",
      },
      {
        en: "AskUserQuestion cards: the model asks you a question with choices and waits for the answer.",
        th: "การ์ด AskUserQuestion: โมเดลถามคำถามพร้อมตัวเลือกแล้วรอคำตอบของคุณ",
      },
      {
        en: "Slash commands: /compact, /context, /memory, /init, /rewind, /permissions, /clear — and any imported skill.",
        th: "คำสั่ง slash: /compact, /context, /memory, /init, /rewind, /permissions, /clear — และ skill ที่นำเข้า",
      },
    ],
    evidence: [
      { label: "serve/checkpoints.py · 19 tests", href: file("serve/checkpoints.py") },
      { label: "serve/files.py · 17 tests", href: file("serve/files.py") },
      { label: "serve/test_askq.py · 13 tests", href: file("serve/test_askq.py") },
      { label: "serve/test_steer.py", href: file("serve/test_steer.py") },
    ],
  },
  {
    id: "context",
    category: "see",
    status: "shipped",
    title: { en: "Context you can see and compact", th: "เห็นและย่อ context ได้" },
    summary: {
      en: "A chip on the prompt bar says how much of the window is used. The panel splits it into the conversation, tool calls and results, and instructions.",
      th: "ชิปบนแถบพิมพ์บอกว่าใช้ context ไปเท่าไร แผงแยกเป็นบทสนทนา การเรียกเครื่องมือและผลลัพธ์ และคำสั่ง",
    },
    points: [
      {
        en: "/compact summarises the conversation; it also compacts by itself at 95 % of the window.",
        th: "/compact สรุปบทสนทนา และย่ออัตโนมัติเมื่อใช้ถึง 95% ของหน้าต่าง",
      },
      {
        en: "While it compacts, a dedicated orb shows it.",
        th: "ระหว่างย่อ จะมี orb เฉพาะแสดงให้เห็น",
      },
    ],
    scenes: [
      Sc("context", {
        en: "Context window: 5,820 of 262,144 tokens, split by part, with the automatic compaction point at 249,036 tokens (95 %).",
        th: "Context window: 5,820 จาก 262,144 โทเคน แยกตามส่วน และจุดย่ออัตโนมัติที่ 249,036 โทเคน (95%)",
      }),
    ],
    evidence: [
      { label: "lib/compact.ts", href: file("serve/ui/src/lib/compact.ts") },
      { label: "lib/context.ts", href: file("serve/ui/src/lib/context.ts") },
      { label: "lib/compact.test.ts", href: file("serve/ui/src/lib/compact.test.ts") },
    ],
  },
  {
    id: "monitor",
    category: "see",
    status: "shipped",
    title: { en: "See the machine", th: "เห็นเครื่อง" },
    summary: {
      en: "Dashboard, Live, Requests and Hardware, plus a Trace viewer for pipeline timelines. A figure the engine does not measure says so.",
      th: "Dashboard, Live, Requests และ Hardware พร้อมตัวดู Trace สำหรับ timeline ของ pipeline ตัวเลขที่เอนจินไม่ได้วัดจะบอกว่าไม่ได้วัด",
    },
    points: [
      {
        en: "“Where the last request's experts ran”: primary GPU, secondary GPU, RAM over PCIe, RAM on the CPU.",
        th: "“expert ของคำขอล่าสุดทำงานที่ไหน”: GPU หลัก, GPU รอง, RAM ผ่าน PCIe, RAM บน CPU",
      },
      {
        en: "Requests are kept on this PC with the first 200 characters of each prompt; keep the whole prompt of the next 1, 5 or 10 on demand.",
        th: "คำขอถูกเก็บไว้ในเครื่องนี้พร้อม 200 ตัวอักษรแรกของ prompt และเลือกเก็บ prompt เต็มของ 1, 5 หรือ 10 คำขอถัดไปได้",
      },
      {
        en: "Hardware shows which expert kernels are in use (for example AVX-VNNI), the pool workers, each GPU's load, memory, temperature and link.",
        th: "Hardware แสดงว่าใช้ expert kernel ตัวไหน (เช่น AVX-VNNI) จำนวน pool worker และโหลด หน่วยความจำ อุณหภูมิ และลิงก์ของแต่ละ GPU",
      },
    ],
    scenes: [
      Sc("dashboard", { en: "Dashboard: speed, hardware, requests, and where the last request's experts ran.", th: "Dashboard: ความเร็ว ฮาร์ดแวร์ คำขอ และที่ที่ expert ของคำขอล่าสุดทำงาน" }, { en: "Dashboard", th: "Dashboard" }),
      Sc("live", { en: "Live: decode speed while the model writes, GPU load and recent requests.", th: "Live: ความเร็ว decode ขณะโมเดลเขียน โหลดของ GPU และคำขอล่าสุด" }, { en: "Live", th: "Live" }),
      Sc("requests", { en: "Requests: each request with tokens read, cached, written and speeds.", th: "Requests: แต่ละคำขอพร้อมโทเคนที่อ่าน อยู่ใน cache เขียน และความเร็ว" }, { en: "Requests", th: "Requests" }),
      Sc("hardware", { en: "Hardware: GPUs, CPU expert kernels in use, pool workers, memory.", th: "Hardware: GPU, expert kernel ของ CPU ที่ใช้อยู่, pool worker, หน่วยความจำ" }, { en: "Hardware", th: "Hardware" }),
    ],
    note: {
      en: "The screens are rebuilt from a real session on the developer's PC, with a throwaway demo project. Figures such as RAM in use describe the whole PC at that moment.",
      th: "ภาพเป็นเซสชันจริงบนเครื่องของนักพัฒนากับโปรเจกต์ตัวอย่างที่สร้างขึ้นเพื่อการนี้ ตัวเลขเช่น RAM ที่ใช้ เป็นของทั้งเครื่อง ณ ขณะนั้น",
    },
    evidence: [
      { label: "serve/telemetry.py · 11 tests", href: file("serve/telemetry.py") },
      { label: "serve/history.py · 25 tests", href: file("serve/history.py") },
      { label: "pages/Dashboard.tsx", href: file("serve/ui/src/pages/Dashboard.tsx") },
      { label: "pages/Trace.tsx", href: file("serve/ui/src/pages/Trace.tsx") },
    ],
  },
  {
    id: "thai",
    category: "see",
    status: "shipped",
    title: { en: "Thai, all the way", th: "ภาษาไทยครบทั้งระบบ" },
    summary: {
      en: "The whole app is available in Thai with an EN / ไทย switch, and the model side was tuned for Thai too.",
      th: "ทั้งแอปมีภาษาไทยพร้อมปุ่มสลับ EN / ไทย และฝั่งโมเดลก็ปรับให้เหมาะกับภาษาไทยด้วย",
    },
    points: [
      {
        en: "Thai tokens are in the speculative draft head's vocabulary, so Thai answers can draft ahead; the default draft subset is Thai + English/code.",
        th: "โทเคนภาษาไทยอยู่ในคำศัพท์ของ draft head แบบ speculative ทำให้คำตอบไทยร่างล่วงหน้าได้ ค่าเริ่มต้นเป็นชุดไทย + อังกฤษ/โค้ด",
      },
      {
        en: "A loop guard stops a degenerate text loop, with a shorter threshold for Thai.",
        th: "ตัวกันวนลูปหยุดข้อความที่วนซ้ำผิดปกติ โดยใช้เกณฑ์ที่สั้นกว่าสำหรับภาษาไทย",
      },
    ],
    scenes: [
      Sc(
        "dashboard",
        {
          en: "The Dashboard in Thai: the same page, with the app's own Thai words.",
          th: "Dashboard ภาษาไทย: หน้าเดียวกัน ด้วยคำภาษาไทยของแอปเอง",
        },
        undefined,
        "th",
      ),
    ],
    evidence: [
      { label: "serve/ui/src/i18n/th.ts", href: file("serve/ui/src/i18n/th.ts") },
      { label: "serve/loop_guard.py", href: file("serve/loop_guard.py") },
      { label: "tests/xeno/test_draft_vocab.py", href: file("tests/xeno/test_draft_vocab.py") },
      { label: "Issue #55", href: issue(55) },
    ],
  },
  {
    id: "compat",
    category: "api",
    status: "shipped",
    title: { en: "Extends the Anthropic API for Claude Code", th: "ขยาย Anthropic API สำหรับ Claude Code" },
    summary: {
      en: "Point Claude Code at Strata with ANTHROPIC_BASE_URL. The server handles what a coding agent sends.",
      th: "ชี้ Claude Code มาที่ Strata ด้วย ANTHROPIC_BASE_URL เซิร์ฟเวอร์รับมือสิ่งที่ coding agent ส่งมาได้",
    },
    points: [
      {
        en: "stop_sequences, signature_delta on thinking blocks, disable_parallel_tool_use, and a thinking budget that closes the block when it is spent.",
        th: "stop_sequences, signature_delta ของ thinking block, disable_parallel_tool_use และงบ thinking ที่ปิด block เมื่อใช้หมด",
      },
      {
        en: "PDF `document` blocks, and documents or images inside a `tool_result`, reach the model as text pages or images.",
        th: "บล็อก `document` แบบ PDF และเอกสารหรือรูปภายใน `tool_result` ไปถึงโมเดลในรูปหน้าข้อความหรือรูปภาพ",
      },
      {
        en: "Claude Code's billing-header line is removed from the system prompt so the prompt stays identical for caching.",
        th: "บรรทัด billing-header ของ Claude Code ถูกตัดออกจาก system prompt เพื่อให้ prompt เหมือนเดิมสำหรับ cache",
      },
      {
        en: "While the model loads, the port answers 503 / Anthropic 529 overloaded_error, so Claude Code retries instead of backing off.",
        th: "ระหว่างโหลดโมเดล พอร์ตตอบ 503 / Anthropic 529 overloaded_error เพื่อให้ Claude Code ลองใหม่แทนที่จะถอย",
      },
      {
        en: "Side requests (titles, the auto-mode judge) use their own cache slot, so they do not evict the main conversation's.",
        th: "คำขอรอง (ชื่อแชต ตัวตัดสินของ auto-mode) ใช้ cache slot แยก จึงไม่ไล่ของบทสนทนาหลักออก",
      },
    ],
    note: {
      en: "Strata already spoke /v1/messages, count_tokens and thinking blocks; these are the additions. Verified by unit tests; a live check was made once, on a development build with Claude Code 2.1.285.",
      th: "Strata รองรับ /v1/messages, count_tokens และ thinking block อยู่แล้ว สิ่งข้างบนคือส่วนที่เพิ่ม ตรวจด้วย unit test และทดลองใช้งานจริงหนึ่งครั้งบน dev build กับ Claude Code 2.1.285",
    },
    evidence: [
      { label: "serve/frontend.py", href: file("serve/frontend.py") },
      { label: "serve/pdf_blocks.py", href: file("serve/pdf_blocks.py") },
      { label: "test_billing_header.py", href: file("tests/xeno/test_billing_header.py") },
      { label: "test_document_blocks.py", href: file("tests/xeno/test_document_blocks.py") },
      { label: "test_anthropic_stream.py", href: file("tests/xeno/test_anthropic_stream.py") },
    ],
  },
  {
    id: "dynamic-experts",
    chart: "ram",
    category: "engine",
    status: "shipped",
    title: { en: "Dynamic experts: RAM follows VRAM", th: "Dynamic experts: RAM ลดตาม VRAM" },
    summary: {
      en: "A GPU that owns an expert keeps no copy of it in system RAM. The more VRAM is free, the less RAM the model needs.",
      th: "GPU ที่ถือ expert จะไม่เก็บสำเนาไว้ใน RAM ของระบบ ยิ่งมี VRAM ว่างมาก โมเดลก็ยิ่งใช้ RAM น้อยลง",
    },
    points: [
      {
        en: "Exclusive GPU ownership: after an expert is copied into a GPU cache slot and byte-verified, its host pages are decommitted.",
        th: "GPU เป็นเจ้าของ expert แต่ผู้เดียว: หลังคัดลอก expert เข้า cache slot ของ GPU และตรวจทีละไบต์แล้ว หน้า RAM ของมันจะถูกคืน",
      },
      {
        en: "Placement-first cold start: the arena reserves address space for all experts but commits nothing; GPU tiers fill straight from the pack.",
        th: "เริ่มจากการวางตำแหน่ง: arena จองพื้นที่ที่อยู่ของทุก expert แต่ยังไม่ commit และชั้น GPU เติมตรงจาก pack",
      },
      {
        en: "Paired adaptive swaps: the hottest CPU-served experts swap with the coldest GPU-owned ones, so the GPUs follow the conversation without a duplicate host copy.",
        th: "สลับแบบคู่: expert ที่ CPU ให้บริการและร้อนที่สุดสลับกับ expert ที่ GPU ถือและเย็นที่สุด GPU จึงตามบทสนทนาได้โดยไม่มีสำเนาใน RAM",
      },
      {
        en: "The VRAM cache is sized from what is free, minus a reserve, and the KV cache for your context is reserved first.",
        th: "ขนาด cache ใน VRAM คำนวณจากส่วนที่ว่างลบส่วนสำรอง และจอง KV cache สำหรับ context ของคุณก่อน",
      },
    ],
    note: {
      en: "Measured numbers and their conditions are in the Numbers section. Windows and NVIDIA validated; the second-GPU tier is written for an RTX 5060 Ti + RTX 4070 SUPER pair.",
      th: "ตัวเลขที่วัดและเงื่อนไขอยู่ในส่วน Numbers ตรวจสอบบน Windows และ NVIDIA ชั้น GPU ใบที่สองเขียนไว้สำหรับคู่ RTX 5060 Ti + RTX 4070 SUPER",
    },
    evidence: [
      { label: "src/program/generate.cpp", href: file("src/program/generate.cpp") },
      { label: "exclusive_host_pages.cpp", href: file("tests/xeno/exclusive_host_pages.cpp") },
      { label: "placement_formats.cpp", href: file("tests/xeno/placement_formats.cpp") },
      { label: "Issue #4", href: issue(4) },
      { label: "Issue #45", href: issue(45) },
    ],
  },
  {
    id: "capacity",
    chart: "capacity",
    category: "engine",
    status: "opt-in",
    title: { en: "Capacity mode: models larger than RAM", th: "Capacity mode: โมเดลที่ใหญ่กว่า RAM" },
    summary: {
      en: "Experts beyond the GPU tiers live in a bounded RAM cache backed by NVMe, read on demand. It is for models that do not fit.",
      th: "expert ที่เกินชั้น GPU อยู่ใน cache RAM ที่จำกัดขนาด โดยมี NVMe หนุนหลังและอ่านเมื่อต้องการ ใช้กับโมเดลที่ใส่ไม่พอดี",
    },
    points: [
      {
        en: "Unbuffered, overlapped reads (DirectFile), a per-size slab, decayed-LFU eviction with a per-layer index, and an aligned expert pack so a miss is one read straight into its slot.",
        th: "อ่านแบบ unbuffered และซ้อนกับงานอื่น (DirectFile), slab ตามขนาด, ตัดทิ้งแบบ decayed-LFU พร้อม index ต่อ layer และ expert pack ที่จัดแนวไว้ เพื่อให้ miss คืออ่านครั้งเดียวเข้า slot ตรงๆ",
      },
      {
        en: "Optional mirror on a second NVMe: reads are split between the disks.",
        th: "ใส่ mirror บน NVMe ลูกที่สองได้ การอ่านจะแบ่งไปทั้งสองดิสก์",
      },
      {
        en: "Ran IQ3_XXS (about 40 GiB of experts) on a 48 GB PC with identical greedy output at every cache size.",
        th: "รัน IQ3_XXS (expert ราว 40 GiB) บนเครื่อง 48 GB ได้ โดยผลลัพธ์ greedy เหมือนกันทุกขนาด cache",
      },
    ],
    note: {
      en: "Measured on that setup (IQ3_XXS, both GPUs, 512 tokens): decode rises about 51–52 % when the RAM cache goes from 6 to 12 GiB. No comparison with other low-RAM modes is recorded, so none is claimed.",
      th: "วัดบนชุดนั้น (IQ3_XXS, ทั้งสอง GPU, 512 โทเคน): decode เพิ่มราว 51–52% เมื่อ cache RAM จาก 6 เป็น 12 GiB ยังไม่มีบันทึกการเทียบกับโหมด low-RAM อื่น จึงไม่ได้อ้าง",
    },
    evidence: [
      { label: "src/core/expert_source.cpp", href: file("src/core/expert_source.cpp") },
      { label: "nvme_capacity.cpp", href: file("tests/xeno/nvme_capacity.cpp") },
      { label: "PR #60", href: pr(60) },
      { label: "PR #63", href: pr(63) },
      { label: "PR #98", href: pr(98) },
      { label: "Issue #11", href: issue(11) },
    ],
  },
  {
    id: "prefill",
    category: "engine",
    status: "opt-in",
    title: { en: "Two GPUs reading one prompt", th: "สอง GPU อ่าน prompt เดียวกัน" },
    summary: {
      en: "While a prompt is read, the second GPU runs the experts it owns (the expert split), and two lanes overlap one lane's trunk with the other's experts (the wave).",
      th: "ขณะอ่าน prompt GPU ใบที่สองรัน expert ที่มันถือ (expert split) และสองเลนซ้อนกัน โดยลำตัวของเลนหนึ่งทำงานพร้อม expert ของอีกเลน (wave)",
    },
    points: [
      {
        en: "Runs per layer; layers that do not fit the split stay on the first card.",
        th: "ทำงานต่อ layer โดย layer ที่ไม่เหมาะกับการแบ่งจะอยู่บนการ์ดใบแรก",
      },
      {
        en: "Switched on with STRATA_PREFILL_EXPERT_SPLIT and STRATA_PREFILL_WAVE; STRATA_PREFILL_SPLIT_MIN sets the smallest chunk that splits. All three are on in the developer's daily profile.",
        th: "เปิดด้วย STRATA_PREFILL_EXPERT_SPLIT และ STRATA_PREFILL_WAVE และ STRATA_PREFILL_SPLIT_MIN กำหนดชิ้นเล็กสุดที่แบ่ง ทั้งสามเปิดอยู่ในโปรไฟล์ประจำวันของนักพัฒนา",
      },
    ],
    note: {
      en: "Measured only against the fork's own earlier configuration, so no speed-up over upstream is claimed here.",
      th: "วัดเทียบกับค่าตั้งก่อนหน้าของ fork เองเท่านั้น จึงไม่ได้อ้างว่าเร็วกว่า upstream",
    },
    evidence: [
      { label: "src/prefill/prefill.cpp", href: file("src/prefill/prefill.cpp") },
      { label: "src/prefill/split_plan.cpp", href: file("src/prefill/split_plan.cpp") },
      { label: "PR #114", href: pr(114) },
      { label: "PR #116", href: pr(116) },
      { label: "PR #120", href: pr(120) },
    ],
  },
  {
    id: "diagnostics",
    category: "engine",
    status: "opt-in",
    title: { en: "Where did the time go? A timeline", th: "เวลาหายไปไหน? timeline" },
    summary: {
      en: "One switch records every thread and GPU lane of a run on one clock; one command turns it into a budget or a Perfetto trace.",
      th: "สวิตช์เดียวบันทึกทุก thread และเลน GPU ของการรันไว้บนนาฬิกาเดียว และคำสั่งเดียวแปลงเป็นงบเวลาหรือ trace ของ Perfetto",
    },
    points: [
      {
        en: "STRATA_TIMELINE=run.json, then python tests/xeno/perf/timeline.py run.json (add --perfetto to open it in ui.perfetto.dev).",
        th: "STRATA_TIMELINE=run.json แล้วรัน python tests/xeno/perf/timeline.py run.json (เพิ่ม --perfetto เพื่อเปิดใน ui.perfetto.dev)",
      },
      {
        en: "Answers: GPU time by phase and layer, why the copy engine idled, decode ms per round of every stage, pool workers' wake-ups, server requests from HTTP to first token.",
        th: "ตอบได้ว่า: เวลา GPU แยกตามเฟสและ layer ทำไม copy engine ว่าง ms ต่อรอบ decode ของทุกขั้น การตื่นของ pool worker และคำขอของเซิร์ฟเวอร์ตั้งแต่ HTTP ถึงโทเคนแรก",
      },
      {
        en: "The project's rule: record a timeline before adding ad-hoc timers, and claim a speed-up only from paired ABBA runs with identical output.",
        th: "กฎของโปรเจกต์: บันทึก timeline ก่อนเพิ่มตัวจับเวลาชั่วคราว และอ้างว่าเร็วขึ้นได้จากการรันสลับ ABBA ที่ผลลัพธ์เหมือนกันเท่านั้น",
      },
    ],
    note: {
      en: "Costs about 6 % of an 8K prompt read (same-session ABBA), so it is off by default.",
      th: "กินเวลาราว 6% ของการอ่าน prompt 8K (ABBA ในเซสชันเดียวกัน) จึงปิดไว้เป็นค่าเริ่มต้น",
    },
    evidence: [
      { label: "src/platform/timeline.cpp", href: file("src/platform/timeline.cpp") },
      { label: "tests/xeno/perf/timeline.py", href: file("tests/xeno/perf/timeline.py") },
      { label: "docs/reports/2026-09-29-pipeline-timeline.md", href: file("docs/reports/2026-09-29-pipeline-timeline.md") },
      { label: "Issue #33", href: issue(33) },
    ],
  },
  {
    id: "kernel",
    category: "engine",
    status: "shipped",
    title: { en: "Low-level work: kernels and the system", th: "งานระดับล่าง: kernel และระบบ" },
    summary: {
      en: "CPU, GPU and Windows details that decide how a model on a normal PC behaves.",
      th: "รายละเอียดของ CPU, GPU และ Windows ที่กำหนดว่าโมเดลบนพีซีทั่วไปจะทำงานอย่างไร",
    },
    points: [
      {
        en: "AVX-VNNI path for Q2_0 expert rows (vpdpbusd), bit-exact with the AVX2 path, dispatch AVX-512 → VNNI → AVX2. Kernel-level +7–13 %; whole-engine decode is within noise.",
        th: "เส้นทาง AVX-VNNI สำหรับแถว expert แบบ Q2_0 (vpdpbusd) ตรงกับ AVX2 ทุกบิต ส่งงาน AVX-512 → VNNI → AVX2 เร็วขึ้น +7–13% ระดับ kernel ส่วน decode ทั้งเอนจินอยู่ในช่วง noise",
      },
      {
        en: "A second-GPU expert tier with one CUDA graph per layer, paired swaps and a free-VRAM floor so the display card keeps headroom.",
        th: "ชั้น expert ของ GPU ใบที่สองที่ใช้ CUDA graph เดียวต่อ layer สลับแบบคู่ และเพดาน VRAM ว่างขั้นต่ำเพื่อให้การ์ดจอแสดงผลมีที่เหลือ",
      },
      {
        en: "CPU pool: thread priority, rest between verify windows instead of spinning, core order by efficiency class.",
        th: "CPU pool: ลำดับความสำคัญของ thread พักระหว่างรอบ verify แทนการวนรอ และเรียงคอร์ตามระดับประสิทธิภาพ",
      },
      {
        en: "A Windows crash reporter writes a symbolised stack and a minidump into the engine log.",
        th: "ตัวรายงานการล่มบน Windows เขียน stack ที่แปลงชื่อแล้วและ minidump ลงใน log ของเอนจิน",
      },
      {
        en: "An engine watch restarts an engine that died by itself, and gives a degraded engine its second-GPU tier back when the card is idle.",
        th: "ตัวเฝ้าเอนจินสตาร์ทเอนจินที่ล่มเองใหม่ และคืนชั้น GPU ใบที่สองให้เอนจินที่ถูกลดระดับเมื่อการ์ดว่าง",
      },
    ],
    note: {
      en: "Whole-engine effects below the project's 13.6 % noise gate are treated as unproved, and the AVX-VNNI note above says so.",
      th: "ผลระดับทั้งเอนจินที่ต่ำกว่าเกณฑ์ noise 13.6% ของโปรเจกต์ถือว่ายังพิสูจน์ไม่ได้ และหมายเหตุของ AVX-VNNI ข้างบนก็ระบุไว้แล้ว",
    },
    evidence: [
      { label: "q2_avx2.cpp", href: file("src/kernels/cpu/q2_avx2.cpp") },
      { label: "kernels/cpu/expert.hpp", href: file("include/strata/kernels/cpu/expert.hpp") },
      { label: "src/platform/crash_report.cpp", href: file("src/platform/crash_report.cpp") },
      { label: "src/core/secondary_runner.cpp", href: file("src/core/secondary_runner.cpp") },
    ],
  },
];
