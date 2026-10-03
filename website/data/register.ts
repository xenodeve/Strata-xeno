import type { L } from "@/lib/i18n";
import type { Status } from "./features";
import { file, issue, pr } from "./sources";

export type RegCategory = "app" | "agent" | "setup" | "api" | "engine" | "tools";

export type Row = {
  id: string;
  cat: RegCategory;
  title: L;
  benefit: L;
  status: Status;
  /** Files at the stamped revision. */
  where: { label: string; href: string }[];
  tests?: string;
  /** Conditions or limits that belong to this row. */
  note?: L;
};

export const REG_CATEGORIES: { id: RegCategory; label: L }[] = [
  { id: "app", label: { en: "Web app", th: "เว็บแอป" } },
  { id: "agent", label: { en: "Coding agent", th: "Coding agent" } },
  { id: "setup", label: { en: "Your setup", th: "สิ่งที่คุณมีอยู่" } },
  { id: "api", label: { en: "Server & API", th: "เซิร์ฟเวอร์และ API" } },
  { id: "engine", label: { en: "Engine", th: "เอนจิน" } },
  { id: "tools", label: { en: "Measuring & tooling", th: "การวัดและเครื่องมือ" } },
];

const w = (label: string, path: string) => ({ label, href: file(path) });
const ui = (name: string) => w(name, `serve/ui/src/${name}`);

export const REGISTER: Row[] = [
  // ── Web app ────────────────────────────────────────────────
  {
    id: "next-app", cat: "app", status: "shipped",
    title: { en: "A new web app at /", th: "เว็บแอปใหม่ที่ /" },
    benefit: { en: "Chat, Dashboard, Live, Requests, Hardware, Settings and About in one app: 7 pages, 4 sub-pages, 35 components. The classic app stays at /classic/.", th: "Chat, Dashboard, Live, Requests, Hardware, Settings และ About ในแอปเดียว: 7 หน้า 4 หน้าย่อย 35 component ส่วนแอปคลาสสิกยังอยู่ที่ /classic/" },
    where: [w("serve/ui/", "serve/ui/package.json"), w("serve/test_ui.py", "serve/test_ui.py"), { label: "PR #104", href: pr(104) }],
    tests: "bun test · 450, e2e · 56, test_ui.py · 11",
    note: { en: "Merged on main; the switch of / to the new app still awaits the developer's confirmation (#110).", th: "merge เข้า main แล้ว การสลับให้ / เปิดแอปใหม่ยังรอการยืนยันจากนักพัฒนา (#110)" },
  },
  {
    id: "monitor-pages", cat: "app", status: "shipped",
    title: { en: "Dashboard, Live, Requests, Hardware, Trace", th: "Dashboard, Live, Requests, Hardware, Trace" },
    benefit: { en: "Per-request timings with stall attribution, per-GPU load, memory, temperature and link, the disks behind the model, and a viewer for pipeline timelines.", th: "เวลาต่อคำขอพร้อมการระบุสาเหตุที่ติดขัด โหลด หน่วยความจำ อุณหภูมิ และลิงก์ของแต่ละ GPU ดิสก์ที่เก็บโมเดล และตัวดู timeline ของ pipeline" },
    where: [ui("pages/Dashboard.tsx"), ui("pages/Requests.tsx"), ui("pages/Hardware.tsx"), ui("pages/Trace.tsx"), w("serve/telemetry.py", "serve/telemetry.py")],
    tests: "test_telemetry.py · 11",
  },
  {
    id: "thai-ui", cat: "app", status: "shipped",
    title: { en: "Full Thai interface", th: "หน้าตาภาษาไทยครบ" },
    benefit: { en: "An EN / ไทย switch; the Thai strings live in their own files per area.", th: "ปุ่มสลับ EN / ไทย ข้อความไทยแยกไฟล์ตามส่วนของแอป" },
    where: [w("i18n/th.ts", "serve/ui/src/i18n/th.ts"), w("lib/i18n.test.ts", "serve/ui/src/lib/i18n.test.ts")],
  },
  {
    id: "marks", cat: "app", status: "shipped",
    title: { en: "Status marks: orbs, loaders, 18 avatars", th: "เครื่องหมายสถานะ: orb, ตัวโหลด, avatar 18 แบบ" },
    benefit: { en: "Pick how \"the model is working\" looks; each status has its own form, and avatars hop, look around or sleep.", th: "เลือกหน้าตาของ “โมเดลกำลังทำงาน” แต่ละสถานะมีรูปแบบของตัวเอง และ avatar กระโดด เหลียวมอง หรือหลับได้" },
    where: [ui("lib/orbs.ts"), ui("lib/avatar.ts"), ui("components/StatusMarks.tsx")],
    tests: "avatar.test.ts, orbclock.test.ts",
  },
  {
    id: "status-words", cat: "app", status: "shipped",
    title: { en: "A word-based status that never goes quiet", th: "สถานะเป็นคำที่ไม่เคยเงียบ" },
    benefit: { en: "One status for what is happening now, from writing a call to reading its result, with Thai wording; a new step must add its own words.", th: "สถานะเดียวสำหรับสิ่งที่เกิดขึ้นตอนนี้ ตั้งแต่เขียนคำสั่งจนถึงอ่านผล มีคำภาษาไทย และขั้นใหม่ต้องเพิ่มคำของตัวเอง" },
    where: [ui("lib/status.ts"), w("AGENTS.md", "AGENTS.md")],
    tests: "status.test.ts",
  },
  {
    id: "settings", cat: "app", status: "shipped",
    title: { en: "Settings in 12 topics", th: "Settings 12 หัวข้อ" },
    benefit: { en: "Status marks, API key, Coding tools, Permissions, Hooks, Web access, Sub-agents, MCP Servers, Limits, and Import (Skills, MCP servers, Memory).", th: "Status marks, API key, Coding tools, Permissions, Hooks, Web access, Sub-agents, MCP Servers, Limits และ Import (Skills, MCP servers, Memory)" },
    where: [ui("pages/Settings.tsx"), ui("components/PermissionsSettings.tsx"), ui("components/McpSettings.tsx")],
  },
  // ── Coding agent ───────────────────────────────────────────
  {
    id: "tools", cat: "agent", status: "shipped",
    title: { en: "15 coding tools with Claude Code's names", th: "เครื่องมือเขียนโค้ด 15 ตัวที่ใช้ชื่อแบบ Claude Code" },
    benefit: { en: "Read, Write, Edit, Glob, Grep, TodoWrite, ExitPlanMode, NotebookEdit, AskUserQuestion, Bash, BashOutput, KillShell, WebFetch, WebSearch, Task, as a built-in server of the MCP hub.", th: "Read, Write, Edit, Glob, Grep, TodoWrite, ExitPlanMode, NotebookEdit, AskUserQuestion, Bash, BashOutput, KillShell, WebFetch, WebSearch, Task ในรูปเซิร์ฟเวอร์ในตัวของ MCP hub" },
    where: [w("serve/agent.py", "serve/agent.py"), w("serve/shell.py", "serve/shell.py")],
    tests: "test_agent.py · 71, test_agent_chat.py · 32, test_shell.py · 29",
  },
  {
    id: "permissions", cat: "agent", status: "shipped",
    title: { en: "A permission system in Claude Code's rule syntax", th: "ระบบสิทธิ์ด้วยไวยากรณ์กฎแบบ Claude Code" },
    benefit: { en: "Deny beats allow beats defaults; modes Ask, Plan, Auto; secrets and .git writes are never auto-allowed.", th: "deny ชนะ allow ชนะค่าเริ่มต้น มีโหมด Ask, Plan, Auto และไม่มีการอนุญาตอัตโนมัติสำหรับความลับและการเขียน .git" },
    where: [w("serve/permissions.py", "serve/permissions.py")],
    tests: "test_permissions.py · 44",
  },
  {
    id: "judge", cat: "agent", status: "shipped",
    title: { en: "Auto mode's second opinion", th: "ความเห็นที่สองของ Auto mode" },
    benefit: { en: "A no-tools side request rates each call 1–5 for safety before auto-approving; it never sees tool output.", th: "คำขอรองที่ไม่มีเครื่องมือให้คะแนนความปลอดภัย 1–5 ก่อนอนุมัติอัตโนมัติ และไม่เคยเห็นผลลัพธ์ของเครื่องมือ" },
    where: [w("serve/judge.py", "serve/judge.py"), w("serve/agent_run.py", "serve/agent_run.py")],
    tests: "test_agent_run.py · 21",
  },
  {
    id: "hooks", cat: "agent", status: "shipped",
    title: { en: "Hooks, written by hand", th: "Hook ที่เขียนด้วยมือ" },
    benefit: { en: "Your own commands before or after a tool call, on a prompt, and on stop; exit code 2 blocks a call. Configured only in the run config, never from a web page.", th: "คำสั่งของคุณเองก่อน/หลังเรียกเครื่องมือ เมื่อส่งข้อความ และเมื่อจบ exit code 2 จะบล็อกการเรียก ตั้งค่าได้เฉพาะใน run config ไม่ใช่จากหน้าเว็บ" },
    where: [w("serve/hooks.py", "serve/hooks.py")],
    tests: "test_hooks.py · 46",
    note: { en: "None are defined by default.", th: "ไม่มี hook ถูกตั้งไว้เป็นค่าเริ่มต้น" },
  },
  {
    id: "subagents", cat: "agent", status: "off-by-default",
    title: { en: "Sub-agents", th: "Sub-agent" },
    benefit: { en: "The Task tool hands a side task to a helper with its own conversation, which answers with a short report and keeps long searches out of the chat.", th: "เครื่องมือ Task ส่งงานข้างเคียงให้ตัวช่วยที่มีบทสนทนาของตัวเอง แล้วตอบกลับเป็นรายงานสั้น ช่วยเก็บการค้นหายาวๆ ออกจากแชต" },
    where: [w("serve/subagent.py", "serve/subagent.py")],
    tests: "test_subagent.py · 24",
    note: { en: "Off until switched on: a helper uses the same model, so the main answer waits.", th: "ปิดจนกว่าจะเปิด: ตัวช่วยใช้โมเดลเดียวกัน คำตอบหลักจึงต้องรอ" },
  },
  {
    id: "web", cat: "agent", status: "off-by-default",
    title: { en: "Web access with a question every time", th: "เข้าเว็บโดยถามทุกครั้ง" },
    benefit: { en: "WebFetch and WebSearch (Brave or SearXNG). Every call shows what would be sent; addresses on this PC or your network are refused; page text is treated as text, never as an order.", th: "WebFetch และ WebSearch (Brave หรือ SearXNG) ทุกครั้งจะแสดงสิ่งที่จะถูกส่ง ปฏิเสธที่อยู่ในเครื่องนี้หรือเครือข่ายของคุณ และถือเนื้อหาเว็บเป็นข้อความ ไม่ใช่คำสั่ง" },
    where: [w("serve/web.py", "serve/web.py")],
    tests: "test_web.py · 53",
    note: { en: "Not tried with a real model before the merge (PR #104).", th: "ยังไม่ได้ลองกับโมเดลจริงก่อน merge (PR #104)" },
  },
  {
    id: "media", cat: "agent", status: "shipped",
    title: { en: "Read images and PDFs", th: "อ่านรูปและ PDF" },
    benefit: { en: "Read on an image goes to the model when vision is loaded; a PDF is read by page range, scans as images.", th: "Read กับรูปจะส่งให้โมเดลเมื่อโหลด vision ไว้ ส่วน PDF อ่านเป็นช่วงหน้า และหน้าสแกนส่งเป็นรูป" },
    where: [w("serve/media.py", "serve/media.py")],
    tests: "test_media.py · 24",
    note: { en: "Not tried with a real model before the merge (PR #104).", th: "ยังไม่ได้ลองกับโมเดลจริงก่อน merge (PR #104)" },
  },
  {
    id: "rewind", cat: "agent", status: "shipped",
    title: { en: "Rewind", th: "Rewind" },
    benefit: { en: "Every prompt is a checkpoint; files the tools changed are put back, files someone else changed are left alone.", th: "ทุกข้อความคือ checkpoint ไฟล์ที่เครื่องมือแก้จะถูกคืนค่า ส่วนไฟล์ที่คนอื่นแก้จะไม่ถูกแตะ" },
    where: [w("serve/checkpoints.py", "serve/checkpoints.py")],
    tests: "test_checkpoints.py · 19",
  },
  {
    id: "projects", cat: "agent", status: "shipped",
    title: { en: "Projects with several folders", th: "โปรเจกต์หลายโฟลเดอร์" },
    benefit: { en: "One or more folders per project (worktrees of one repo), a main folder, per-project permission rules and chats listed under the project.", th: "หนึ่งโฟลเดอร์หรือมากกว่าต่อโปรเจกต์ (worktree ของ repo เดียวกัน) มีโฟลเดอร์หลัก กฎสิทธิ์ต่อโปรเจกต์ และแชตแสดงใต้โปรเจกต์" },
    where: [w("serve/folders.py", "serve/folders.py"), ui("components/NewProjectDialog.tsx")],
    tests: "test_folders.py · 10",
  },
  {
    id: "git-panel", cat: "agent", status: "shipped",
    title: { en: "Git, Plan, Skills, Memory, Context panel", th: "แผง Git, Plan, Skills, Memory, Context" },
    benefit: { en: "Read-only git state with diffs, the model's plan, skills, the notes it was handed, and the context window beside the chat.", th: "สถานะ git แบบอ่านอย่างเดียวพร้อม diff แผนของโมเดล skills โน้ตที่โมเดลได้รับ และ context window ข้างแชต" },
    where: [w("serve/gitview.py", "serve/gitview.py"), ui("components/SidePanel.tsx")],
    tests: "test_gitview.py · 23",
  },
  {
    id: "controls", cat: "agent", status: "shipped",
    title: { en: "@file, message queue, steering, runs that survive a refresh", th: "@file, คิวข้อความ, steer, run ที่รอดการรีเฟรช" },
    benefit: { en: "Mention a file with @; send a message while the agent works and it is read at the next step; stop a run; refresh the page and re-attach to the running answer.", th: "อ้างไฟล์ด้วย @ ส่งข้อความระหว่างที่ agent ทำงานแล้วถูกอ่านในขั้นถัดไป หยุด run ได้ และรีเฟรชหน้าแล้วกลับมาต่อกับคำตอบที่กำลังรัน" },
    where: [w("serve/runs.py", "serve/runs.py"), w("serve/files.py", "serve/files.py")],
    tests: "test_runs.py · 21, test_steer.py · 7, test_files.py · 17",
  },
  {
    id: "askq", cat: "agent", status: "shipped",
    title: { en: "AskUserQuestion cards", th: "การ์ด AskUserQuestion" },
    benefit: { en: "The model asks a question with choices and waits for your answer in the page.", th: "โมเดลถามคำถามพร้อมตัวเลือกและรอคำตอบของคุณในหน้าเว็บ" },
    where: [w("serve/agent_run.py", "serve/agent_run.py")],
    tests: "test_askq.py · 13",
  },
  {
    id: "compact", cat: "agent", status: "shipped",
    title: { en: "/compact, /context and automatic compaction", th: "/compact, /context และการย่ออัตโนมัติ" },
    benefit: { en: "A context chip and panel; compaction by command or by itself at 95 % of the window.", th: "ชิปและแผง context ย่อบทสนทนาด้วยคำสั่ง หรืออัตโนมัติเมื่อใช้ถึง 95% ของหน้าต่าง" },
    where: [ui("lib/compact.ts"), ui("lib/context.ts"), ui("components/ContextPanel.tsx")],
    tests: "compact.test.ts, context.test.ts",
  },
  {
    id: "project-rules", cat: "agent", status: "shipped",
    title: { en: "Project instructions are read", th: "อ่านคำสั่งของโปรเจกต์" },
    benefit: { en: "The project's CLAUDE.md and AGENTS.md (and .claude/rules, @include) are handed to the model after the rules.", th: "CLAUDE.md และ AGENTS.md ของโปรเจกต์ (รวม .claude/rules และ @include) ถูกส่งให้โมเดลต่อจากกฎ" },
    where: [w("serve/agent_prompt.py", "serve/agent_prompt.py"), w("serve/memory.py", "serve/memory.py")],
    tests: "test_agent_prompt.py · 11, test_memory.py · 22",
  },
  // ── Your setup ─────────────────────────────────────────────
  {
    id: "import-skills", cat: "setup", status: "shipped",
    title: { en: "Import skills from your other coding apps", th: "นำ skill จากแอปเขียนโค้ดอื่นของคุณมาใช้" },
    benefit: { en: "Finds skills in Claude Code, Codex, Antigravity, Gemini CLI, Cursor and the shared ~/.agents folder, and lets the chat load one when a task fits.", th: "ค้นหา skill ใน Claude Code, Codex, Antigravity, Gemini CLI, Cursor และโฟลเดอร์ ~/.agents ที่ใช้ร่วมกัน และให้แชตโหลดมาใช้เมื่องานเข้ากัน" },
    where: [w("serve/harness.py", "serve/harness.py"), w("serve/skills.py", "serve/skills.py")],
    tests: "test_harness.py · 78",
    note: { en: "Found skills are on until you switch an app off in Settings › Import.", th: "skill ที่พบเปิดใช้อยู่จนกว่าคุณจะปิดแอปนั้นใน Settings › Import" },
  },
  {
    id: "import-mcp", cat: "setup", status: "shipped",
    title: { en: "Import MCP servers, or set them up in the page", th: "นำเข้า MCP server หรือตั้งค่าในหน้าเว็บ" },
    benefit: { en: "One Import per server copies it into Strata's own list, where you can edit, turn off or delete it; add one, or paste from Claude Desktop. Secrets are masked and the config is backed up before a write.", th: "กด Import ต่อหนึ่ง server เพื่อคัดลอกเข้ารายการของ Strata ซึ่งแก้ ปิด หรือลบได้ เพิ่มเอง หรือวางจาก Claude Desktop ความลับถูกปิดบังและสำรอง config ก่อนเขียน" },
    where: [w("serve/mcp_admin.py", "serve/mcp_admin.py"), w("serve/harness.py", "serve/harness.py")],
    tests: "test_mcp_admin.py · 16",
    note: { en: "Nothing from another app runs until you import it.", th: "ไม่มีอะไรจากแอปอื่นรันจนกว่าคุณจะนำเข้า" },
  },
  {
    id: "import-memory", cat: "setup", status: "shipped",
    title: { en: "Read the memory your other apps already keep", th: "อ่าน memory ที่แอปอื่นของคุณเก็บไว้อยู่แล้ว" },
    benefit: { en: "Claude Code's and Codex's instruction files and Claude Code's per-project memory can be read by the chat. Each is off until you switch it on, and none changes what the chat may do.", th: "ไฟล์คำสั่งของ Claude Code และ Codex และ memory ต่อโปรเจกต์ของ Claude Code อ่านโดยแชตได้ แต่ละอันปิดอยู่จนกว่าจะเปิด และไม่เปลี่ยนสิ่งที่แชตทำได้" },
    where: [w("serve/memory.py", "serve/memory.py")],
    tests: "test_memory.py · 22",
  },
  // ── Server & API ───────────────────────────────────────────
  {
    id: "anthropic", cat: "api", status: "shipped",
    title: { en: "Anthropic Messages API additions for Claude Code", th: "ส่วนเสริม Anthropic Messages API สำหรับ Claude Code" },
    benefit: { en: "stop_sequences, signature_delta, disable_parallel_tool_use, thinking-budget close, PDF document blocks, documents or images inside tool_result.", th: "stop_sequences, signature_delta, disable_parallel_tool_use, การปิด thinking เมื่อหมดงบ บล็อกเอกสาร PDF และเอกสารหรือรูปใน tool_result" },
    where: [w("serve/frontend.py", "serve/frontend.py"), w("serve/pdf_blocks.py", "serve/pdf_blocks.py"), w("serve/think_budget.py", "serve/think_budget.py")],
    tests: "test_document_blocks.py · 4, test_anthropic_stream.py · 4",
    note: { en: "Upstream already had /v1/messages, count_tokens and thinking blocks.", th: "upstream มี /v1/messages, count_tokens และ thinking block อยู่แล้ว" },
  },
  {
    id: "billing-header", cat: "api", status: "shipped",
    title: { en: "Claude Code's billing header stripped", th: "ตัด billing header ของ Claude Code" },
    benefit: { en: "The x-anthropic-billing-header line is removed from the system prompt so the prompt stays byte-identical and the cache keeps working.", th: "ตัดบรรทัด x-anthropic-billing-header ออกจาก system prompt เพื่อให้ prompt ตรงกันทุกไบต์และ cache ใช้ต่อได้" },
    where: [w("serve/frontend.py", "serve/frontend.py")],
    tests: "test_billing_header.py · 4",
  },
  {
    id: "loading-503", cat: "api", status: "shipped",
    title: { en: "Answers 503 / 529 while the model loads", th: "ตอบ 503 / 529 ระหว่างโหลดโมเดล" },
    benefit: { en: "A placeholder server holds the port while the model loads and answers 503 (Anthropic: 529 overloaded_error), so Claude Code retries instead of backing off.", th: "เซิร์ฟเวอร์ชั่วคราวถือพอร์ตไว้ระหว่างโหลดและตอบ 503 (Anthropic: 529 overloaded_error) เพื่อให้ Claude Code ลองใหม่แทนที่จะถอย" },
    where: [w("serve/server.py", "serve/server.py")],
    note: { en: "The test mapping for this one is not confirmed.", th: "ยังไม่ยืนยันว่ามี test ครอบคลุมข้อนี้" },
  },
  {
    id: "gate", cat: "api", status: "shipped",
    title: { en: "Request priority and a cache slot per prompt family", th: "ลำดับความสำคัญของคำขอและ cache slot ต่อกลุ่ม prompt" },
    benefit: { en: "One engine request at a time with the streamed main turn first; side requests use their own slot so they do not evict the main conversation's cache.", th: "ทีละคำขอต่อเอนจินโดยเทิร์นหลักที่สตรีมอยู่มาก่อน คำขอรองใช้ slot ของตัวเองจึงไม่ไล่ cache ของบทสนทนาหลัก" },
    where: [w("serve/server.py", "serve/server.py")],
    tests: "test_server.py · 221",
  },
  {
    id: "history", cat: "api", status: "shipped",
    title: { en: "Request history on disk", th: "ประวัติคำขอบนดิสก์" },
    benefit: { en: "Monthly JSONL summaries kept forever and a size-capped gzip detail, behind the Requests page.", th: "สรุปแบบ JSONL รายเดือนเก็บตลอด และรายละเอียดแบบ gzip ที่จำกัดขนาด ใช้กับหน้า Requests" },
    where: [w("serve/history.py", "serve/history.py")],
    tests: "test_history.py · 25",
  },
  {
    id: "guards", cat: "api", status: "shipped",
    title: { en: "Loop guard and thinking budget", th: "ตัวกันวนลูปและงบ thinking" },
    benefit: { en: "Cancels a degenerate text loop (Thai after 64 characters, others after 512) and closes the thinking block when its budget is spent.", th: "ยกเลิกข้อความที่วนซ้ำผิดปกติ (ไทยหลัง 64 ตัวอักษร ภาษาอื่นหลัง 512) และปิด thinking เมื่อหมดงบ" },
    where: [w("serve/loop_guard.py", "serve/loop_guard.py"), w("serve/think_budget.py", "serve/think_budget.py")],
    tests: "test_loop_guard.py · 4",
  },
  {
    id: "cjk", cat: "api", status: "opt-in",
    title: { en: "CJK guard", th: "CJK guard" },
    benefit: { en: "Bans Han tokens when the turn contains none and does not ask for Chinese, so a Thai or English answer does not drift into Chinese.", th: "ห้ามโทเคนอักษรฮั่นเมื่อเทิร์นไม่มีตัวอักษรฮั่นและไม่ได้ขอภาษาจีน เพื่อกันไม่ให้คำตอบไทยหรืออังกฤษเลยไปเป็นจีน" },
    where: [w("serve/cjk_guard.py", "serve/cjk_guard.py")],
    note: { en: "Switch on with \"cjk_guard\": true in the run config.", th: "เปิดด้วย \"cjk_guard\": true ใน run config" },
  },
  // ── Engine ─────────────────────────────────────────────────
  {
    id: "exclusive", cat: "engine", status: "shipped",
    title: { en: "Exclusive GPU ownership and placement-first start", th: "GPU ถือ expert แต่ผู้เดียว และเริ่มจากการวางตำแหน่ง" },
    benefit: { en: "An expert a GPU owns has no host copy; the model starts without ever putting those experts in RAM.", th: "expert ที่ GPU ถือไม่มีสำเนาใน RAM และโมเดลเริ่มโดยไม่เคยนำ expert เหล่านั้นเข้า RAM" },
    where: [w("src/program/generate.cpp", "src/program/generate.cpp"), w("placement_formats.hpp", "include/strata/core/placement_formats.hpp")],
    tests: "exclusive_host_pages.cpp, placement_formats.cpp",
    note: { en: "Windows only (page decommit). Default on for all-Q2_0 packs; a flag turns it on for any native pack.", th: "ใช้ได้บน Windows เท่านั้น (คืนหน้า RAM) เปิดเป็นค่าเริ่มต้นกับ pack Q2_0 ทั้งหมด และมี flag เปิดให้ pack native อื่นๆ" },
  },
  {
    id: "second-gpu", cat: "engine", status: "opt-in",
    title: { en: "A second-GPU expert tier", th: "ชั้น expert ของ GPU ใบที่สอง" },
    benefit: { en: "The second card holds experts and runs their rows in decode, as one CUDA graph per layer, with paired swaps and a free-VRAM floor for the display card.", th: "การ์ดใบที่สองถือ expert และรันแถวของมันตอน decode เป็น CUDA graph เดียวต่อ layer สลับแบบคู่ และมีเพดาน VRAM ว่างขั้นต่ำสำหรับการ์ดจอแสดงผล" },
    where: [w("secondary_runner.cpp", "src/core/secondary_runner.cpp"), w("secondary_arena.cpp", "src/core/secondary_arena.cpp")],
    tests: "secondary_arena.cpp, secondary_budget.cpp, secondary_iq_parity.cpp",
    note: { en: "Written for an RTX 5060 Ti + RTX 4070 SUPER pair (sm_89).", th: "เขียนไว้สำหรับคู่ RTX 5060 Ti + RTX 4070 SUPER (sm_89)" },
  },
  {
    id: "capacity", cat: "engine", status: "opt-in",
    title: { en: "Capacity mode", th: "Capacity mode" },
    benefit: { en: "Bounded RAM cache with NVMe behind it for models that do not fit; DirectFile reads, slab, eviction index, aligned pack, optional mirror.", th: "cache RAM จำกัดขนาดมี NVMe หนุนหลังสำหรับโมเดลที่ใส่ไม่พอดี อ่านแบบ DirectFile มี slab, index การตัดทิ้ง, aligned pack และ mirror ได้" },
    where: [w("expert_source.cpp", "src/core/expert_source.cpp"), w("nvme_capacity.cpp", "tests/xeno/nvme_capacity.cpp")],
    tests: "nvme_capacity, nvme_slots, nvme_evict_index, nvme_overlap, nvme_mirror, nvme_aligned",
    note: { en: "Opt-in: it trades decode time for RAM.", th: "เปิดเองเมื่อต้องการ: แลก เวลา decode กับ RAM" },
  },
  {
    id: "split-wave", cat: "engine", status: "opt-in",
    title: { en: "Expert split and wave for prompt reading", th: "Expert split และ wave สำหรับการอ่าน prompt" },
    benefit: { en: "The second GPU runs the experts it owns during a prompt, and two lanes overlap trunk and experts.", th: "GPU ใบที่สองรัน expert ที่มันถือระหว่างอ่าน prompt และสองเลนซ้อนกันระหว่างลำตัวและ expert" },
    where: [w("prefill.cpp", "src/prefill/prefill.cpp"), w("split_plan.cpp", "src/prefill/split_plan.cpp")],
    tests: "split_plan.cpp",
  },
  {
    id: "tail", cat: "engine", status: "shipped",
    title: { en: "A tail file for borrowed cache slots", th: "tail file สำหรับ cache slot ที่ยืมมา" },
    benefit: { en: "Cache slots the prompt path borrows are refilled from a ~3.5 GB file instead of host copies, saving RAM.", th: "cache slot ที่เส้นทาง prompt ยืมมาถูกเติมคืนจากไฟล์ราว 3.5 GB แทนสำเนาใน RAM จึงประหยัด RAM" },
    where: [w("src/program/generate.cpp", "src/program/generate.cpp")],
    note: { en: "Default on; costs about 3.5 GB of disk. Private commit 41.16 → 37.90 GiB on an 8K prompt (Q2_0, same session, identical output).", th: "เปิดเป็นค่าเริ่มต้น ใช้ดิสก์ราว 3.5 GB private commit 41.16 → 37.90 GiB ที่ prompt 8K (Q2_0, เซสชันเดียวกัน, ผลลัพธ์เหมือนกัน)" },
  },
  {
    id: "vnni", cat: "engine", status: "shipped",
    title: { en: "AVX-VNNI Q2_0 CPU expert rows", th: "AVX-VNNI สำหรับแถว expert Q2_0 บน CPU" },
    benefit: { en: "vpdpbusd path, bit-exact with AVX2, dispatch AVX-512 → VNNI → AVX2; STRATA_FORCE_AVX2=1 turns it off.", th: "เส้นทาง vpdpbusd ตรงกับ AVX2 ทุกบิต ส่งงาน AVX-512 → VNNI → AVX2 และ STRATA_FORCE_AVX2=1 ปิดได้" },
    where: [w("q2_avx2.cpp", "src/kernels/cpu/q2_avx2.cpp"), w("expert.hpp", "include/strata/kernels/cpu/expert.hpp")],
    tests: "xeno_q2_isa_parity",
    note: { en: "+7–13 % at kernel level; whole-engine decode is within noise.", th: "เร็วขึ้น +7–13% ระดับ kernel ส่วน decode ทั้งเอนจินอยู่ในช่วง noise" },
  },
  {
    id: "pool", cat: "engine", status: "shipped",
    title: { en: "CPU pool that rests, with priority and core order", th: "CPU pool ที่พักได้ มีลำดับความสำคัญและลำดับคอร์" },
    benefit: { en: "Workers sleep between verify windows instead of spinning 20 ms, run at high thread priority, and are placed by efficiency class.", th: "worker พักระหว่างรอบ verify แทนการวนรอ 20 ms ทำงานด้วยลำดับความสำคัญของ thread สูง และวางตามระดับประสิทธิภาพของคอร์" },
    where: [w("generate.cpp", "src/program/generate.cpp")],
    tests: "pool_rest.cpp, pool_core_policy.cpp, pool_spin.cpp",
  },
  {
    id: "crash", cat: "engine", status: "shipped",
    title: { en: "A crash reporter", th: "ตัวรายงานการล่ม" },
    benefit: { en: "A symbolised stack and a minidump (strata-crash-*.dmp) in the engine log when the engine crashes.", th: "stack ที่แปลงชื่อแล้วและ minidump (strata-crash-*.dmp) ใน log ของเอนจินเมื่อเอนจินล่ม" },
    where: [w("crash_report.cpp", "src/platform/crash_report.cpp")],
    tests: "STRATA_TEST_CRASH=1",
    note: { en: "Windows / MSVC only.", th: "ใช้ได้บน Windows / MSVC เท่านั้น" },
  },
  {
    id: "watch", cat: "engine", status: "shipped",
    title: { en: "An engine watch", th: "ตัวเฝ้าเอนจิน" },
    benefit: { en: "Restarts an engine that died by itself, and gives a degraded engine its second-GPU tier back when idle and VRAM is free.", th: "สตาร์ทเอนจินที่ล่มเองใหม่ และคืนชั้น GPU ใบที่สองให้เอนจินที่ถูกลดระดับเมื่อว่างและมี VRAM" },
    where: [w("serve/server.py", "serve/server.py")],
    note: { en: "The test mapping for this one is not confirmed.", th: "ยังไม่ยืนยันว่ามี test ครอบคลุมข้อนี้" },
  },
  {
    id: "thai-draft", cat: "engine", status: "shipped",
    title: { en: "Thai tokens in the draft head", th: "โทเคนไทยใน draft head" },
    benefit: { en: "The speculative draft vocabulary includes Thai, so Thai answers can draft ahead; setup's default subset is Thai + English/code.", th: "คำศัพท์ draft แบบ speculative มีภาษาไทย คำตอบไทยจึงร่างล่วงหน้าได้ และชุดเริ่มต้นของ setup คือไทย + อังกฤษ/โค้ด" },
    where: [w("setup.py", "setup.py"), w("test_draft_vocab.py", "tests/xeno/test_draft_vocab.py")],
    tests: "test_draft_vocab.py, test_setup_draft_vocab.py",
    note: { en: "Measured against the fork's own earlier draft vocabulary (issue #55).", th: "วัดเทียบกับ draft vocabulary เดิมของ fork เอง (issue #55)" },
  },
  {
    id: "ple", cat: "engine", status: "shipped",
    title: { en: "PLE read-ahead for decode windows", th: "อ่านล่วงหน้า PLE สำหรับรอบ decode" },
    benefit: { en: "The next window's PLE rows are read as tokens become known (--ple-ahead, default on).", th: "อ่านแถว PLE ของรอบถัดไปทันทีที่รู้โทเคน (--ple-ahead เปิดเป็นค่าเริ่มต้น)" },
    where: [w("generate.cpp", "src/program/generate.cpp")],
    tests: "ple_reader_test.cpp",
    note: { en: "Upstream 0.1.38 independently shipped a first-chunk read-ahead.", th: "upstream 0.1.38 ออกแนวคิดอ่านล่วงหน้าของชิ้นแรกด้วยตัวเอง" },
  },
  // ── Measuring & tooling ────────────────────────────────────
  {
    id: "timeline", cat: "tools", status: "opt-in",
    title: { en: "STRATA_TIMELINE and its analyser", th: "STRATA_TIMELINE และตัววิเคราะห์" },
    benefit: { en: "Every thread and GPU lane on one clock; a budget report and Perfetto export.", th: "ทุก thread และเลน GPU บนนาฬิกาเดียว มีรายงานงบเวลาและส่งออกไป Perfetto" },
    where: [w("timeline.cpp", "src/platform/timeline.cpp"), w("timeline.py", "tests/xeno/perf/timeline.py"), { label: "Issue #33", href: issue(33) }],
    tests: "test_timeline.py · 17",
    note: { en: "About 6 % of an 8K prompt read.", th: "ราว 6% ของการอ่าน prompt 8K" },
  },
  {
    id: "abba", cat: "tools", status: "shipped",
    title: { en: "ABBA runner and per-stage latency parser", th: "ตัวรัน ABBA และตัวแยกวิเคราะห์ latency ต่อขั้น" },
    benefit: { en: "Paired, alternating, profiler-off runs with the same output as the gate for every speed claim; counters parsed by a script, not by eye.", th: "การรันแบบจับคู่สลับกัน ปิด profiler และผลลัพธ์ต้องตรงกัน เป็นเกณฑ์ของทุกคำกล่าวอ้างเรื่องความเร็ว ตัวนับถูกแยกด้วยสคริปต์ ไม่ใช่ด้วยตา" },
    where: [w("AGENTS.md", "AGENTS.md"), w("ab.py", "tests/xeno/perf/ab.py"), w("latency_breakdown.py", "tests/xeno/latency_breakdown.py")],
  },
  {
    id: "route-trace", cat: "tools", status: "shipped",
    title: { en: "Routing trace format 2 and a cache-policy simulator", th: "routing trace รูปแบบ 2 และตัวจำลองนโยบาย cache" },
    benefit: { en: "A commit-aware trace of which experts each token used, with a simulator to test cache policies offline.", th: "trace ที่รู้เรื่อง commit ว่าแต่ละโทเคนใช้ expert ตัวไหน พร้อมตัวจำลองเพื่อทดสอบนโยบาย cache แบบออฟไลน์" },
    where: [w("routing_trace.hpp", "include/strata/core/routing_trace.hpp")],
    tests: "routing_trace.cpp, test_route_tools_format2.py",
  },
  {
    id: "blueprint", cat: "tools", status: "shipped",
    title: { en: "A blueprint checked at commit time", th: "blueprint ที่ตรวจตอน commit" },
    benefit: { en: "docs/BLUEPRINT.md describes the processes, the life of a request, every flag and every environment variable; a commit hook stops a new flag or file without a blueprint update.", th: "docs/BLUEPRINT.md อธิบายโปรเซส เส้นทางของคำขอ ทุก flag และทุก environment variable และ commit hook จะหยุด commit ที่เพิ่ม flag หรือไฟล์ใหม่โดยไม่อัปเดต blueprint" },
    where: [w("docs/BLUEPRINT.md", "docs/BLUEPRINT.md"), w("tools/blueprint_check.py", "tools/blueprint_check.py")],
    note: { en: "The blueprint's baseline commit lags behind main; its \"not yet in the baseline\" table lists the difference.", th: "commit ฐานของ blueprint ตามหลัง main อยู่ ตาราง “ยังไม่อยู่ในฐาน” ระบุส่วนต่างไว้" },
  },
  {
    id: "mock", cat: "tools", status: "shipped",
    title: { en: "Develop the UI without a GPU", th: "พัฒนา UI โดยไม่ต้องมี GPU" },
    benefit: { en: "A mock server runs the real routes, tools and pages with a scripted model, so UI work needs no model and never touches the daily server.", th: "mock server รัน route เครื่องมือ และหน้าจริงด้วยโมเดลสคริปต์ งาน UI จึงไม่ต้องมีโมเดลและไม่แตะเซิร์ฟเวอร์ประจำวัน" },
    where: [w("mock_server.py", "serve/ui/dev/mock_server.py")],
    tests: "e2e checks · 56",
  },
];
