// Thai for the timeline (trace) viewer page. span, lane, window, STRATA_TIMELINE stay in Latin.
export const part: Record<string, string> = {
  "There is no span in this file.": "ไม่มี span ในไฟล์นี้",
  "This is not a trace file (a Chrome trace array, as STRATA_TIMELINE writes).": "นี่ไม่ใช่ไฟล์ trace (อาร์เรย์ Chrome trace แบบที่ STRATA_TIMELINE เขียน)",
  "Timeline": "ไทม์ไลน์",
  "The engine's pipeline timeline: start the engine with {cmd}, then open the file here. It is read in this browser and goes nowhere. Scroll to zoom, drag to move, double-click to see it all.":
    "ไทม์ไลน์ของ pipeline ใน engine: เริ่ม engine ด้วย {cmd} แล้วเปิดไฟล์ที่นี่ ไฟล์ถูกอ่านในเบราว์เซอร์นี้และไม่ถูกส่งไปที่ใด เลื่อนล้อเมาส์เพื่อซูม ลากเพื่อเลื่อน ดับเบิลคลิกเพื่อดูทั้งหมด",
  "Reading…": "กำลังอ่าน…",
  "Open a timeline file": "เปิดไฟล์ timeline",
  "spans": "span",
  "lanes": "lane",
  "Find a span name": "ค้นหาชื่อ span",
  "Download for Perfetto": "ดาวน์โหลดสำหรับ Perfetto",
  "Reading the file…": "กำลังอ่านไฟล์…",
  "Drop a timeline file here, or open one.": "วางไฟล์ timeline ที่นี่ หรือเปิดไฟล์",
  "Timeline of {n} lanes; the table below lists what the visible window spent its time on": "ไทม์ไลน์ {n} lane ตารางด้านล่างแสดงว่า window ที่มองเห็นใช้เวลาไปกับอะไร",
  "{lane} · {dur} · at {at}": "{lane} · {dur} · ที่ {at}",
  "In the window": "ใน window",
  "Lanes run in parallel, so these totals can add up to more than the window.": "lane ทำงานขนานกัน ผลรวมเหล่านี้จึงอาจมากกว่า window",
  "{c} span": "{c} span",
  "{c} spans": "{c} span",
}
