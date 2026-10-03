import type { BotAvatarState, BotAvatarType } from "@/components/Avatar";
import type { SceneId } from "@/components/scenes";
import type { L } from "@/lib/i18n";

export type Slide = {
  id: string;
  /** The orb form of the app that this avatar stands for (lib/orbs.ts in serve/ui). */
  status: string;
  avatar: BotAvatarType;
  state: BotAvatarState;
  title: L;
  line: L;
  /** The piece of the app's interface that shows it, played as a scene, and the sentence a screen reader gets for it. */
  scene?: { id: SceneId; alt: L };
};

/**
 * The nine moods of the app. The app tells what is going on by the form of a mark, and the avatars are its
 * marks: the shape for each status below is the app's own choice (lib/avatar.ts, "By status").
 */
export const SLIDES: Slide[] = [
  {
    id: "listening",
    status: "Listening",
    avatar: "cloud",
    state: "default",
    title: { en: "It listens.", th: "มันรับฟัง" },
    line: { en: "Your prompt is read on your PC. Nothing leaves it.", th: "prompt ถูกอ่านบนพีซีของคุณ ไม่มีอะไรออกไปข้างนอก" },
    scene: { id: "listening", alt: { en: "The empty chat's prompt bar: “Nothing leaves it.”", th: "แถบพิมพ์ของแชตว่าง: “Nothing leaves it.”" } },
  },
  {
    id: "solving",
    status: "Solving",
    avatar: "cat",
    state: "working",
    title: { en: "It thinks, in words.", th: "มันคิด เป็นคำพูด" },
    line: { en: "Thinking shows as words and a count, never a bare spinner.", th: "การคิดแสดงเป็นคำและตัวเลข ไม่ใช่แค่วงกลมหมุน" },
    scene: { id: "thinking", alt: { en: "The Live page while the model thinks: “Thinking · 452 tokens”, with the speed.", th: "หน้า Live ขณะโมเดลคิด: “Thinking · 452 tokens” พร้อมความเร็ว" } },
  },
  {
    id: "searching",
    status: "Searching",
    avatar: "droid",
    state: "default",
    title: { en: "It looks around.", th: "มันมองหา" },
    line: { en: "A lookup, or an idle server: the droid looks around.", th: "การค้นหา หรือเซิร์ฟเวอร์ที่ว่างอยู่: droid เหลียวมองไปรอบๆ" },
    scene: { id: "ready", alt: { en: "The Dashboard at rest: “Ready”.", th: "Dashboard ตอนว่าง: “Ready”" } },
  },
  {
    id: "connecting",
    status: "Connecting",
    avatar: "flower",
    state: "working",
    title: { en: "It reaches for a tool.", th: "มันหยิบเครื่องมือ" },
    line: { en: "Edit, Bash, Read: every call is a card.", th: "Edit, Bash, Read: ทุกการเรียกเป็นการ์ดหนึ่งใบ" },
    scene: { id: "tools", alt: { en: "Edit and Bash cards in the chat, each with its time.", th: "การ์ด Edit และ Bash ในแชต พร้อมเวลาของแต่ละอัน" } },
  },
  {
    id: "working",
    status: "Working",
    avatar: "mech",
    state: "working",
    title: { en: "It works.", th: "มันทำงาน" },
    line: { en: "Your GPUs and CPU, shown plainly.", th: "GPU และ CPU ของคุณ แสดงตรงๆ" },
    scene: { id: "gpus", alt: { en: "The Hardware page: two GPUs, the CPU and its expert kernel.", th: "หน้า Hardware: GPU สองใบ CPU และ expert kernel ของมัน" } },
  },
  {
    id: "weaving",
    status: "Weaving",
    avatar: "star",
    state: "working",
    title: { en: "It plans.", th: "มันวางแผน" },
    line: { en: "A multi-step plan becomes a live checklist.", th: "แผนหลายขั้นกลายเป็นเช็กลิสต์สด" },
    scene: { id: "plan", alt: { en: "A to-do list written by the model: “0 of 3 steps done”.", th: "รายการสิ่งที่ต้องทำที่โมเดลเขียน: “0 of 3 steps done”" } },
  },
  {
    id: "composing",
    status: "Composing",
    avatar: "pill",
    state: "working",
    title: { en: "It answers.", th: "มันตอบ" },
    line: { en: "Speed and tokens sit under every reply.", th: "ความเร็วและจำนวนโทเคนอยู่ใต้ทุกคำตอบ" },
    scene: { id: "answer", alt: { en: "A reply with code and “642 tokens · 63.1 tok/s · 3 tool calls”.", th: "คำตอบพร้อมโค้ดและ “642 tokens · 63.1 tok/s · 3 tool calls”" } },
  },
  {
    id: "breathing",
    status: "Breathing",
    avatar: "circle",
    state: "default",
    title: { en: "It waits for you.", th: "มันรอคุณ" },
    line: { en: "A card asks before a command runs. You decide.", th: "การ์ดถามก่อนรันคำสั่ง คุณเป็นคนตัดสินใจ" },
    scene: { id: "permission", alt: { en: "A permission card: “Run this command?” with Allow and Deny.", th: "การ์ดขออนุญาต: “Run this command?” พร้อมปุ่ม Allow และ Deny" } },
  },
  {
    id: "shaping",
    status: "Shaping",
    avatar: "triangle",
    state: "sleeping",
    title: { en: "It sleeps.", th: "มันหลับ" },
    line: { en: "Unload the model and the avatar sleeps.", th: "ถอนโมเดลออก แล้ว avatar ก็หลับ" },
  },
];
