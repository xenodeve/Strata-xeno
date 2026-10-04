import type { IconName } from "@/components/details/icons";
import type { L } from "@/lib/i18n";

/** The details page is laid out like the app's own Settings: groups of topics in a tree, one panel at a time. */
export type Topic = {
  id: string;
  icon: IconName;
  label: L;
  /** What the panel shows: a feature (data/features.ts), a measured figure (data/numbers.ts) or a fixed panel. */
  kind: "feature" | "measured" | "counts" | "register" | "credits";
};

export type Group = { id: string; label: L; topics: Topic[] };

export const GROUPS: Group[] = [
  {
    id: "app",
    label: { en: "The app", th: "แอป" },
    topics: [
      { id: "chat-tools", icon: "terminal", label: { en: "Coding tools", th: "เครื่องมือโค้ด" }, kind: "feature" },
      { id: "ask-first", icon: "shield", label: { en: "Ask first", th: "ถามก่อน" }, kind: "feature" },
      { id: "rules", icon: "key", label: { en: "Permissions", th: "สิทธิ์" }, kind: "feature" },
      { id: "projects", icon: "folder", label: { en: "Projects", th: "โปรเจกต์" }, kind: "feature" },
      { id: "panel", icon: "git", label: { en: "Side panel", th: "แผงข้าง" }, kind: "feature" },
      { id: "controls", icon: "keys", label: { en: "Chat controls", th: "ตัวควบคุมแชต" }, kind: "feature" },
    ],
  },
  {
    id: "setup",
    label: { en: "Your setup", th: "สิ่งที่คุณมี" },
    topics: [{ id: "import", icon: "plug", label: { en: "Import", th: "นำเข้า" }, kind: "feature" }],
  },
  {
    id: "see",
    label: { en: "See what happens", th: "เห็นสิ่งที่เกิดขึ้น" },
    topics: [
      { id: "status", icon: "pulse", label: { en: "Status", th: "สถานะ" }, kind: "feature" },
      { id: "marks", icon: "face", label: { en: "Marks", th: "Marks" }, kind: "feature" },
      { id: "context", icon: "layers", label: { en: "Context", th: "Context" }, kind: "feature" },
      { id: "monitor", icon: "gauge", label: { en: "Monitor", th: "Monitor" }, kind: "feature" },
      { id: "thai", icon: "th", label: { en: "Thai", th: "ภาษาไทย" }, kind: "feature" },
    ],
  },
  {
    id: "api",
    label: { en: "API", th: "API" },
    topics: [{ id: "compat", icon: "link", label: { en: "Claude Code & Anthropic", th: "Claude Code และ Anthropic" }, kind: "feature" }],
  },
  {
    id: "engine",
    label: { en: "Engine", th: "เอนจิน" },
    topics: [
      { id: "dynamic-experts", icon: "chip", label: { en: "Dynamic experts", th: "Dynamic experts" }, kind: "feature" },
      { id: "capacity", icon: "drive", label: { en: "Capacity mode", th: "Capacity mode" }, kind: "feature" },
      { id: "prefill", icon: "split", label: { en: "Two-GPU prompts", th: "prompt สอง GPU" }, kind: "feature" },
      { id: "diagnostics", icon: "timer", label: { en: "Timeline", th: "Timeline" }, kind: "feature" },
      { id: "kernel", icon: "bolt", label: { en: "Kernels & system", th: "Kernel และระบบ" }, kind: "feature" },
    ],
  },
  {
    id: "numbers",
    label: { en: "Numbers", th: "ตัวเลข" },
    topics: [
      { id: "v038-prompt", icon: "timer", label: { en: "vs v0.1.38: prompt", th: "เทียบ v0.1.38: prompt" }, kind: "measured" },
      { id: "v038-decode", icon: "bolt", label: { en: "vs v0.1.38: decode", th: "เทียบ v0.1.38: decode" }, kind: "measured" },
      { id: "v038-ram", icon: "bars", label: { en: "vs v0.1.38: RAM", th: "เทียบ v0.1.38: RAM" }, kind: "measured" },
      { id: "ram", icon: "bars", label: { en: "RAM", th: "RAM" }, kind: "measured" },
      { id: "decode", icon: "bolt", label: { en: "Decode speed", th: "ความเร็ว decode" }, kind: "measured" },
      { id: "counts", icon: "list", label: { en: "Counted", th: "นับจากโค้ด" }, kind: "counts" },
    ],
  },
  {
    id: "more",
    label: { en: "More", th: "เพิ่มเติม" },
    topics: [
      { id: "register", icon: "list", label: { en: "Everything it adds", th: "ทุกสิ่งที่เพิ่ม" }, kind: "register" },
      { id: "credits", icon: "book", label: { en: "Credits & method", th: "เครดิตและวิธีวัด" }, kind: "credits" },
    ],
  },
];

export const ALL_TOPICS = GROUPS.flatMap((g) => g.topics);
